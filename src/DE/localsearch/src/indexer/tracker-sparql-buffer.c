/*
 * Copyright (C) 2011, Nokia <ivan.frade@nokia.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 *
 * Author: Carlos Garnacho <carlos@lanedo.com>
 */

#include "config-miners.h"

#include "tracker-sparql-buffer.h"

#include <tracker-common.h>

#include "tracker-utils.h"

#define DEFAULT_GRAPH "tracker:FileSystem"

typedef struct _TrackerSparqlBufferPrivate TrackerSparqlBufferPrivate;
typedef struct _SparqlTaskData SparqlTaskData;
typedef struct _UpdateBatchData UpdateBatchData;

enum {
	PROP_0,
	PROP_CONNECTION
};

struct _TrackerSparqlBufferPrivate
{
	TrackerSparqlConnection *connection;
	GPtrArray *tasks;
	gint n_updates;
	TrackerBatch *batch;

	TrackerSparqlStatement *delete_file;
	TrackerSparqlStatement *delete_file_content;
	TrackerSparqlStatement *delete_content;
	TrackerSparqlStatement *move_file;
	TrackerSparqlStatement *move_content;
};

enum {
	TASK_TYPE_RESOURCE,
	TASK_TYPE_STMT,
};

struct _SparqlTaskData
{
	guint type;

	union {
		struct {
			gchar *graph;
			TrackerResource *resource;
		} resource;
		struct {
			TrackerSparqlStatement *stmt;
		} stmt;
		struct {
			gchar *sparql;
		} sparql;
	} d;
};

struct _UpdateBatchData {
	TrackerSparqlBuffer *buffer;
	GPtrArray *tasks;
	TrackerBatch *batch;
	GTask *async_task;
};

G_DEFINE_TYPE_WITH_PRIVATE (TrackerSparqlBuffer, tracker_sparql_buffer, TRACKER_TYPE_TASK_POOL)

static void
tracker_sparql_buffer_finalize (GObject *object)
{
	TrackerSparqlBufferPrivate *priv;

	priv = tracker_sparql_buffer_get_instance_private (TRACKER_SPARQL_BUFFER (object));

	g_object_unref (priv->delete_file);
	g_object_unref (priv->delete_file_content);
	g_object_unref (priv->delete_content);
	g_object_unref (priv->move_file);
	g_object_unref (priv->move_content);
	g_object_unref (priv->connection);

	G_OBJECT_CLASS (tracker_sparql_buffer_parent_class)->finalize (object);
}

static void
tracker_sparql_buffer_set_property (GObject      *object,
                                    guint         param_id,
                                    const GValue *value,
                                    GParamSpec   *pspec)
{
	TrackerSparqlBufferPrivate *priv;

	priv = tracker_sparql_buffer_get_instance_private (TRACKER_SPARQL_BUFFER (object));

	switch (param_id) {
	case PROP_CONNECTION:
		priv->connection = g_value_dup_object (value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, param_id, pspec);
		break;
	}
}

static void
tracker_sparql_buffer_get_property (GObject    *object,
                                    guint       param_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
	TrackerSparqlBufferPrivate *priv;

	priv = tracker_sparql_buffer_get_instance_private (TRACKER_SPARQL_BUFFER (object));

	switch (param_id) {
	case PROP_CONNECTION:
		g_value_set_object (value,
		                    priv->connection);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, param_id, pspec);
		break;
	}
}

static void
tracker_sparql_buffer_constructed (GObject *object)
{
	TrackerSparqlBufferPrivate *priv;

	priv = tracker_sparql_buffer_get_instance_private (TRACKER_SPARQL_BUFFER (object));

	priv->delete_file =
		tracker_load_statement (priv->connection, "delete-file.rq", NULL);
	priv->delete_file_content =
		tracker_load_statement (priv->connection, "delete-file-content.rq", NULL);
	priv->delete_content =
		tracker_load_statement (priv->connection, "delete-folder-contents.rq", NULL);
	priv->move_file =
		tracker_load_statement (priv->connection, "move-file.rq", NULL);
	priv->move_content =
		tracker_load_statement (priv->connection, "move-folder-contents.rq", NULL);

	G_OBJECT_CLASS (tracker_sparql_buffer_parent_class)->constructed (object);
}

static void
tracker_sparql_buffer_class_init (TrackerSparqlBufferClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS (klass);

	object_class->finalize = tracker_sparql_buffer_finalize;
	object_class->set_property = tracker_sparql_buffer_set_property;
	object_class->get_property = tracker_sparql_buffer_get_property;
	object_class->constructed = tracker_sparql_buffer_constructed;

	g_object_class_install_property (object_class,
	                                 PROP_CONNECTION,
	                                 g_param_spec_object ("connection",
	                                                      "sparql connection",
	                                                      "Sparql Connection",
	                                                      TRACKER_SPARQL_TYPE_CONNECTION,
	                                                      G_PARAM_READWRITE |
	                                                      G_PARAM_CONSTRUCT_ONLY |
	                                                      G_PARAM_STATIC_STRINGS));
}

static void
tracker_sparql_buffer_init (TrackerSparqlBuffer *buffer)
{
}

TrackerSparqlBuffer *
tracker_sparql_buffer_new (TrackerSparqlConnection *connection,
                           guint                    limit)
{
	return g_object_new (TRACKER_TYPE_SPARQL_BUFFER,
	                     "connection", connection,
	                     "limit", limit,
	                     NULL);
}

static void
remove_task_foreach (TrackerTask     *task,
                     TrackerTaskPool *pool)
{
	tracker_task_pool_remove (pool, task);
}

static void
update_batch_data_free (UpdateBatchData *batch_data)
{
	g_object_unref (batch_data->batch);

	g_ptr_array_unref (batch_data->tasks);

	g_clear_object (&batch_data->async_task);

	g_slice_free (UpdateBatchData, batch_data);
}

static void
batch_execute_cb (GObject      *object,
                  GAsyncResult *result,
                  gpointer      user_data)
{
	TrackerSparqlBufferPrivate *priv;
	TrackerSparqlBuffer *buffer;
	GError *error = NULL;
	UpdateBatchData *update_data;

	update_data = user_data;
	buffer = TRACKER_SPARQL_BUFFER (update_data->buffer);
	priv = tracker_sparql_buffer_get_instance_private (buffer);
	priv->n_updates--;

	TRACKER_NOTE (MINER_FS_EVENTS,
	              g_message ("(Sparql buffer) Finished array-update with %u tasks",
	                         update_data->tasks->len));

	if (!tracker_batch_execute_finish (TRACKER_BATCH (object),
	                                   result,
	                                   &error)) {
		g_task_set_task_data (update_data->async_task,
		                      g_ptr_array_ref (update_data->tasks),
		                      (GDestroyNotify) g_ptr_array_unref);
		g_task_return_error (update_data->async_task, error);
	} else {
		g_task_return_pointer (update_data->async_task,
		                       g_ptr_array_ref (update_data->tasks),
		                       (GDestroyNotify) g_ptr_array_unref);
	}

	update_batch_data_free (update_data);
}

gboolean
tracker_sparql_buffer_flush (TrackerSparqlBuffer *buffer,
                             const gchar         *reason,
                             GAsyncReadyCallback  cb,
                             gpointer             user_data)
{
	TrackerSparqlBufferPrivate *priv;
	UpdateBatchData *update_data;

	priv = tracker_sparql_buffer_get_instance_private (buffer);

	if (priv->n_updates > 0) {
		return FALSE;
	}

	if (!priv->tasks ||
	    priv->tasks->len == 0) {
		return FALSE;
	}

	TRACKER_NOTE (MINER_FS_EVENTS, g_message ("Flushing SPARQL buffer, reason: %s", reason));

	update_data = g_slice_new0 (UpdateBatchData);
	update_data->buffer = buffer;
	update_data->tasks = g_ptr_array_ref (priv->tasks);
	update_data->batch = g_object_ref (priv->batch);
	update_data->async_task = g_task_new (buffer, NULL, cb, user_data);

	/* Empty pool, update_data will keep
	 * references to the tasks to keep
	 * these alive.
	 */
	g_ptr_array_unref (priv->tasks);
	priv->tasks = NULL;
	priv->n_updates++;
	g_clear_object (&priv->batch);

	/* While flushing, remove the tasks from the task pool too, so it's
	 * hinted as below limits again.
	 */
	g_ptr_array_foreach (update_data->tasks,
	                     (GFunc) remove_task_foreach,
	                     update_data->buffer);

	tracker_batch_execute_async (update_data->batch,
	                             NULL,
	                             batch_execute_cb,
	                             update_data);
	return TRUE;
}

static void
sparql_buffer_push_to_pool (TrackerSparqlBuffer *buffer,
                            TrackerTask         *task)
{
	TrackerSparqlBufferPrivate *priv;

	priv = tracker_sparql_buffer_get_instance_private (buffer);

	/* Task pool addition increments reference */
	tracker_task_pool_add (TRACKER_TASK_POOL (buffer), task);

	if (!priv->tasks)
		priv->tasks = g_ptr_array_new_with_free_func ((GDestroyNotify) tracker_task_unref);

	/* We add a reference here because we unref when removed from
	 * the GPtrArray. */
	g_ptr_array_add (priv->tasks, tracker_task_ref (task));
}

static TrackerBatch *
tracker_sparql_buffer_get_current_batch (TrackerSparqlBuffer *buffer)
{
	TrackerSparqlBufferPrivate *priv;

	g_return_val_if_fail (TRACKER_IS_SPARQL_BUFFER (buffer), NULL);

	priv = tracker_sparql_buffer_get_instance_private (TRACKER_SPARQL_BUFFER (buffer));

	if (!priv->batch)
		priv->batch = tracker_sparql_connection_create_batch (priv->connection);

	return priv->batch;
}

static SparqlTaskData *
sparql_task_data_new_resource (const gchar     *graph,
                               TrackerResource *resource)
{
	SparqlTaskData *task_data;

	task_data = g_slice_new0 (SparqlTaskData);
	task_data->type = TASK_TYPE_RESOURCE;
	task_data->d.resource.resource = g_object_ref (resource);
	task_data->d.resource.graph = g_strdup (graph);

	return task_data;
}

static SparqlTaskData *
sparql_task_data_new_stmt (TrackerSparqlStatement *stmt)
{
	SparqlTaskData *task_data;

	task_data = g_slice_new0 (SparqlTaskData);
	task_data->type = TASK_TYPE_STMT;
	task_data->d.stmt.stmt = stmt;

	return task_data;
}

static void
sparql_task_data_free (SparqlTaskData *data)
{
	if (data->type == TASK_TYPE_RESOURCE) {
		g_clear_object (&data->d.resource.resource);
		g_free (data->d.resource.graph);
	}

	g_slice_free (SparqlTaskData, data);
}

static void
tracker_sparql_buffer_push (TrackerSparqlBuffer *buffer,
                            GFile               *file,
                            const gchar         *graph,
                            TrackerResource     *resource)
{
	TrackerBatch *batch;
	TrackerTask *task;
	SparqlTaskData *data;

	g_return_if_fail (TRACKER_IS_SPARQL_BUFFER (buffer));
	g_return_if_fail (G_IS_FILE (file));
	g_return_if_fail (TRACKER_IS_RESOURCE (resource));

	batch = tracker_sparql_buffer_get_current_batch (buffer);
	tracker_batch_add_resource (batch, graph, resource);

	data = sparql_task_data_new_resource (graph, resource);

	task = tracker_task_new (file, data,
	                         (GDestroyNotify) sparql_task_data_free);
	sparql_buffer_push_to_pool (buffer, task);
	tracker_task_unref (task);
}

gchar *
tracker_sparql_task_get_sparql (TrackerTask *task)
{
	SparqlTaskData *task_data;

	task_data = tracker_task_get_data (task);

	if (task_data->type == TASK_TYPE_RESOURCE) {
		return tracker_resource_print_sparql_update (task_data->d.resource.resource,
		                                             NULL,
		                                             task_data->d.resource.graph);
	} else if (task_data->type == TASK_TYPE_STMT) {
		return g_strdup (tracker_sparql_statement_get_sparql (task_data->d.stmt.stmt));
	}

	return NULL;
}

GPtrArray *
tracker_sparql_buffer_flush_finish (TrackerSparqlBuffer  *buffer,
                                    GAsyncResult         *res,
                                    GError              **error)
{
	GPtrArray *tasks;

	g_return_val_if_fail (TRACKER_IS_SPARQL_BUFFER (buffer), NULL);
	g_return_val_if_fail (G_IS_ASYNC_RESULT (res), NULL);
	g_return_val_if_fail (!error || !*error, NULL);

	tasks = g_task_propagate_pointer (G_TASK (res), error);

	if (!tasks)
		tasks = g_ptr_array_ref (g_task_get_task_data (G_TASK (res)));

	return tasks;
}

static void
push_stmt_task (TrackerSparqlBuffer    *buffer,
                TrackerSparqlStatement *stmt,
                GFile                  *file)
{
	TrackerTask *task;
	SparqlTaskData *data;

	data = sparql_task_data_new_stmt (stmt);
	task = tracker_task_new (file, data,
	                         (GDestroyNotify) sparql_task_data_free);
	sparql_buffer_push_to_pool (buffer, task);
	tracker_task_unref (task);
}

void
tracker_sparql_buffer_log_delete (TrackerSparqlBuffer *buffer,
                                  GFile               *file)
{
	TrackerSparqlBufferPrivate *priv;
	TrackerBatch *batch;
	g_autofree gchar *uri = NULL;

	g_return_if_fail (TRACKER_IS_SPARQL_BUFFER (buffer));
	g_return_if_fail (G_IS_FILE (file));

	priv = tracker_sparql_buffer_get_instance_private (TRACKER_SPARQL_BUFFER (buffer));

	uri = g_file_get_uri (file);
	batch = tracker_sparql_buffer_get_current_batch (buffer);
	tracker_batch_add_statement (batch, priv->delete_file,
	                             "uri", G_TYPE_STRING, uri,
	                             NULL);
	push_stmt_task (buffer, priv->delete_file, file);
}

void
tracker_sparql_buffer_log_delete_content (TrackerSparqlBuffer *buffer,
                                          GFile               *file)
{
	TrackerSparqlBufferPrivate *priv;
	TrackerBatch *batch;
	g_autofree gchar *uri = NULL;

	g_return_if_fail (TRACKER_IS_SPARQL_BUFFER (buffer));
	g_return_if_fail (G_IS_FILE (file));

	priv = tracker_sparql_buffer_get_instance_private (TRACKER_SPARQL_BUFFER (buffer));

	uri = g_file_get_uri (file);
	batch = tracker_sparql_buffer_get_current_batch (buffer);
	tracker_batch_add_statement (batch, priv->delete_content,
	                             "uri", G_TYPE_STRING, uri,
	                             NULL);
	push_stmt_task (buffer, priv->delete_content, file);
}

void
tracker_sparql_buffer_log_move (TrackerSparqlBuffer *buffer,
                                GFile               *source,
                                GFile               *dest,
                                const gchar         *dest_data_source)
{
	TrackerSparqlBufferPrivate *priv;
	TrackerBatch *batch;
	g_autofree gchar *source_uri = NULL, *dest_uri = NULL, *new_parent_uri = NULL;
	g_autofree gchar *basename = NULL, *path = NULL;
	g_autoptr (GFile) new_parent = NULL;

	g_return_if_fail (TRACKER_IS_SPARQL_BUFFER (buffer));
	g_return_if_fail (G_IS_FILE (source));
	g_return_if_fail (G_IS_FILE (dest));

	priv = tracker_sparql_buffer_get_instance_private (TRACKER_SPARQL_BUFFER (buffer));

	source_uri = g_file_get_uri (source);
	dest_uri = g_file_get_uri (dest);
	path = g_file_get_path (dest);
	new_parent = g_file_get_parent (dest);
	new_parent_uri = g_file_get_uri (new_parent);
	basename = g_filename_display_basename (path);

	batch = tracker_sparql_buffer_get_current_batch (buffer);
	tracker_batch_add_statement (batch, priv->move_file,
	                             "sourceUri", G_TYPE_STRING, source_uri,
	                             "destUri", G_TYPE_STRING, dest_uri,
	                             "newFilename", G_TYPE_STRING, basename,
	                             "newParent", G_TYPE_STRING, new_parent_uri,
	                             "newDataSource", G_TYPE_STRING, dest_data_source,
	                             NULL);

	push_stmt_task (buffer, priv->move_file, dest);
}

void
tracker_sparql_buffer_log_move_content (TrackerSparqlBuffer *buffer,
                                        GFile               *source,
                                        GFile               *dest)
{
	TrackerSparqlBufferPrivate *priv;
	TrackerBatch *batch;
	g_autofree gchar *source_uri = NULL, *dest_uri = NULL;

	g_return_if_fail (TRACKER_IS_SPARQL_BUFFER (buffer));
	g_return_if_fail (G_IS_FILE (source));
	g_return_if_fail (G_IS_FILE (dest));

	priv = tracker_sparql_buffer_get_instance_private (TRACKER_SPARQL_BUFFER (buffer));

	source_uri = g_file_get_uri (source);
	dest_uri = g_file_get_uri (dest);

	batch = tracker_sparql_buffer_get_current_batch (buffer);
	tracker_batch_add_statement (batch, priv->move_content,
	                             "sourceUri", G_TYPE_STRING, source_uri,
	                             "destUri", G_TYPE_STRING, dest_uri,
	                             NULL);

	push_stmt_task (buffer, priv->move_content, dest);
}

void
tracker_sparql_buffer_log_clear_content (TrackerSparqlBuffer *buffer,
                                         GFile               *file)
{
	TrackerSparqlBufferPrivate *priv;
	TrackerBatch *batch;
	g_autofree gchar *uri = NULL;

	g_return_if_fail (TRACKER_IS_SPARQL_BUFFER (buffer));
	g_return_if_fail (G_IS_FILE (file));

	priv = tracker_sparql_buffer_get_instance_private (TRACKER_SPARQL_BUFFER (buffer));
	batch = tracker_sparql_buffer_get_current_batch (buffer);
	uri = g_file_get_uri (file);
	tracker_batch_add_statement (batch, priv->delete_file_content,
	                             "uri", G_TYPE_STRING, uri,
	                             NULL);

	push_stmt_task (buffer, priv->delete_file_content, file);
}

void
tracker_sparql_buffer_log_file (TrackerSparqlBuffer *buffer,
                                GFile               *file,
                                const gchar         *content_graph,
                                TrackerResource     *file_resource,
                                TrackerResource     *graph_resource)
{
	g_autofree gchar *uri = NULL;

	g_return_if_fail (TRACKER_IS_SPARQL_BUFFER (buffer));
	g_return_if_fail (G_IS_FILE (file));
	g_return_if_fail (TRACKER_IS_RESOURCE (file_resource));
	g_return_if_fail (!graph_resource || TRACKER_IS_RESOURCE (graph_resource));

	tracker_sparql_buffer_push (buffer, file, DEFAULT_GRAPH, file_resource);

	if (content_graph && graph_resource)
		tracker_sparql_buffer_push (buffer, file, content_graph, graph_resource);
}

void
tracker_sparql_buffer_log_folder (TrackerSparqlBuffer *buffer,
                                  GFile               *file,
                                  gboolean             is_root,
                                  TrackerResource     *file_resource,
                                  TrackerResource     *folder_resource)
{
	g_autofree gchar *uri = NULL;

	g_return_if_fail (TRACKER_IS_SPARQL_BUFFER (buffer));
	g_return_if_fail (G_IS_FILE (file));
	g_return_if_fail (TRACKER_IS_RESOURCE (file_resource));
	g_return_if_fail (TRACKER_IS_RESOURCE (folder_resource));

	/* Add indexing roots also to content specific graphs to provide the availability information */
	if (is_root) {
		const gchar *special_graphs[] = {
			"tracker:Audio",
			"tracker:Documents",
			"tracker:Pictures",
			"tracker:Software",
			"tracker:Video"
		};
		gint i;

		for (i = 0; i < G_N_ELEMENTS (special_graphs); i++) {
			tracker_sparql_buffer_push (buffer, file, special_graphs[i], folder_resource);
		}
	}

	tracker_sparql_buffer_push (buffer, file, DEFAULT_GRAPH, file_resource);
	tracker_sparql_buffer_push (buffer, file, DEFAULT_GRAPH, folder_resource);
}

void
tracker_sparql_buffer_log_attributes_update (TrackerSparqlBuffer *buffer,
                                             GFile               *file,
                                             const gchar         *content_graph,
                                             TrackerResource     *file_resource,
                                             TrackerResource     *graph_resource)
{
	g_return_if_fail (TRACKER_IS_SPARQL_BUFFER (buffer));
	g_return_if_fail (G_IS_FILE (file));
	g_return_if_fail (TRACKER_IS_RESOURCE (file_resource));
	g_return_if_fail (!graph_resource || TRACKER_IS_RESOURCE (graph_resource));

	if (content_graph && graph_resource)
		tracker_sparql_buffer_push (buffer, file, content_graph, graph_resource);

	tracker_sparql_buffer_push (buffer, file, DEFAULT_GRAPH, file_resource);
}
