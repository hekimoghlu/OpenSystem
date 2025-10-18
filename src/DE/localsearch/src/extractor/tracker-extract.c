/*
 * Copyright (C) 2008, Nokia <ivan.frade@nokia.com>
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
 */

#include "config-miners.h"

#include <glib/gi18n.h>

#include "utils/tracker-extract.h"

#include <valgrind.h>

#include "tracker-extract.h"
#include "tracker-main.h"

G_DEFINE_QUARK (TrackerExtractError, tracker_extract_error)

#define DEFAULT_DEADLINE_SECONDS 5

#define DEFAULT_MAX_TEXT 1048576

static gint deadline_seconds = -1;

typedef struct {
	GTimer *elapsed;
	gint extracted_count;
	gint failed_count;
} StatisticsData;

struct _TrackerExtract {
	GObject parent_instance;

	GHashTable *statistics_data;

	gint max_text;

	GMainContext *thread_context;
	GMainLoop *thread_loop;

	GThread *task_thread;

	GTimer *total_elapsed;

	gint unhandled_count;
};

typedef struct {
	TrackerExtract *extract;
	gchar *content_id;
	gchar *file;
	gchar *mimetype;
	const gchar *graph;
	gint max_text;

	TrackerExtractMetadataFunc func;
	GModule *module;

	GSource *deadline;
} TrackerExtractTaskData;

G_DEFINE_TYPE(TrackerExtract, tracker_extract, G_TYPE_OBJECT)

static void
statistics_data_free (StatisticsData *data)
{
	g_timer_destroy (data->elapsed);
	g_free (data);
}

static void
tracker_extract_init (TrackerExtract *extract)
{
	extract->max_text = DEFAULT_MAX_TEXT;

#ifdef G_ENABLE_DEBUG
	if (TRACKER_DEBUG_CHECK (STATISTICS)) {
		extract->total_elapsed = g_timer_new ();
		g_timer_stop (extract->total_elapsed);
		extract->statistics_data =
			g_hash_table_new_full (NULL, NULL, NULL,
			                       (GDestroyNotify) statistics_data_free);
	}
#endif
}

static void
log_statistics (GObject *object)
{
#ifdef G_ENABLE_DEBUG
	if (TRACKER_DEBUG_CHECK (STATISTICS)) {
		TrackerExtract *extract = TRACKER_EXTRACT (object);
		GHashTableIter iter;
		gpointer key, value;
		gdouble total_elapsed;

		g_message ("--------------------------------------------------");
		g_message ("Statistics:");

		g_hash_table_iter_init (&iter, extract->statistics_data);
		total_elapsed = g_timer_elapsed (extract->total_elapsed, NULL);

		while (g_hash_table_iter_next (&iter, &key, &value)) {
			GModule *module = key;
			StatisticsData *data = value;

			if (data->extracted_count > 0 || data->failed_count > 0) {
				const gchar *name, *name_without_path;

				name = g_module_name (module);
				name_without_path = strrchr (name, G_DIR_SEPARATOR) + 1;

				g_message ("    Module:'%s', extracted:%d, failures:%d, elapsed: %.2fs (%.2f%% of total)",
				           name_without_path,
				           data->extracted_count,
				           data->failed_count,
					   g_timer_elapsed (data->elapsed, NULL),
					   (g_timer_elapsed (data->elapsed, NULL) / total_elapsed) * 100);
			}
		}

		g_message ("Unhandled files: %d", extract->unhandled_count);

		if (extract->unhandled_count == 0 &&
		    g_hash_table_size (extract->statistics_data) < 1) {
			g_message ("    No files handled");
		}

		g_message ("--------------------------------------------------");
	}
#endif
}

static void
tracker_extract_finalize (GObject *object)
{
	TrackerExtract *extract = TRACKER_EXTRACT (object);

	if (extract->thread_loop) {
		g_main_loop_quit (extract->thread_loop);
		g_main_loop_unref (extract->thread_loop);
	}

	if (extract->task_thread)
		g_thread_join (extract->task_thread);

	if (extract->thread_context)
		g_main_context_unref (extract->thread_context);

#ifdef G_ENABLE_DEBUG
	if (TRACKER_DEBUG_CHECK (STATISTICS)) {
		log_statistics (object);
		g_hash_table_destroy (extract->statistics_data);
		g_timer_destroy (extract->total_elapsed);
	}
#endif

	G_OBJECT_CLASS (tracker_extract_parent_class)->finalize (object);
}

static void
tracker_extract_class_init (TrackerExtractClass *klass)
{
	GObjectClass *object_class;

	object_class = G_OBJECT_CLASS (klass);

	object_class->finalize = tracker_extract_finalize;
}

TrackerExtract *
tracker_extract_new (void)
{
	return g_object_new (TRACKER_TYPE_EXTRACT, NULL);
}

static gboolean
get_file_metadata (TrackerExtractTaskData  *task,
                   TrackerExtractInfo     **info_out,
                   GError                 **error)
{
	g_autoptr (TrackerExtractInfo) info = NULL;
	g_autoptr (GFile) file = NULL;
	gboolean success = FALSE;

	*info_out = NULL;

	file = g_file_new_for_uri (task->file);
	info = tracker_extract_info_new (file, task->content_id, task->mimetype, task->graph, task->max_text);

	if (!task->mimetype || !*task->mimetype)
		return FALSE;

	/* Now we have sanity checked everything, actually get the
	 * data we need from the extractors.
	 */
	if (task->func && task->module) {
		g_debug ("Using %s...",
		         g_module_name (task->module));

		success = (task->func) (info, error);
	} else {
		g_autoptr (TrackerResource) resource = NULL;

		/* Dummy extractor */
		resource = tracker_resource_new (NULL);
		tracker_extract_info_set_resource (info, resource);
		success = TRUE;
	}

	if (success)
		*info_out = g_steal_pointer (&info);

	return success;
}

static gboolean
task_deadline_cb (gpointer user_data)
{
	TrackerExtractTaskData *task = user_data;

	g_warning ("File '%s' took too long to process. Shutting down everything",
	           task->file);

	_exit (EXIT_FAILURE);
}

static TrackerExtractTaskData *
extract_task_data_new (TrackerExtract *extract,
                       const gchar    *uri,
                       const gchar    *content_id,
                       const gchar    *mimetype,
                       const gchar    *graph)
{
	TrackerExtractTaskData *task;

	task = g_new0 (TrackerExtractTaskData, 1);
	task->file = g_strdup (uri);
	task->content_id = g_strdup (content_id);
	task->mimetype = g_strdup (mimetype);
	task->graph = graph;
	task->extract = extract;
	task->max_text = extract->max_text;

	task->module = tracker_extract_module_manager_get_module (task->mimetype,
	                                                          NULL,
	                                                          &task->func);

	if (!RUNNING_ON_VALGRIND) {
		if (deadline_seconds < 0) {
			const gchar *deadline_envvar;

			deadline_envvar = g_getenv ("TRACKER_EXTRACT_DEADLINE");
			if (deadline_envvar)
				deadline_seconds = atoi (deadline_envvar);
			else
				deadline_seconds = DEFAULT_DEADLINE_SECONDS;
		}
		task->deadline =
			g_timeout_source_new_seconds (deadline_seconds);
		g_source_set_callback (task->deadline, task_deadline_cb, task, NULL);
		g_source_attach (task->deadline,
		                 g_main_context_get_thread_default ());
	}

	return task;
}

static void
extract_task_data_free (TrackerExtractTaskData *data)
{
	if (data->deadline) {
		g_source_destroy (data->deadline);
		g_source_unref (data->deadline);
	}

	g_free (data->mimetype);
	g_free (data->file);
	g_free (data->content_id);
	g_free (data);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (TrackerExtractTaskData, extract_task_data_free)

static gboolean
get_metadata (GTask *task)
{
	TrackerExtractTaskData *data = g_task_get_task_data (task);
	TrackerExtract *extract = data->extract;
	TrackerExtractInfo *info;
	GError *error = NULL;
	gboolean success = FALSE;

	if (g_task_return_error_if_cancelled (task))
		return FALSE;

#ifdef G_ENABLE_DEBUG
	if (TRACKER_DEBUG_CHECK (STATISTICS)) {
		StatisticsData *stats_data;

		stats_data = g_hash_table_lookup (extract->statistics_data,
						  data->module);
		if (!stats_data) {
			stats_data = g_new0 (StatisticsData, 1);
			stats_data->elapsed = g_timer_new ();
			g_hash_table_insert (extract->statistics_data,
					     data->module,
					     stats_data);
		} else {
			g_timer_continue (stats_data->elapsed);
		}
	}
#endif

	if (get_file_metadata (data, &info, &error)) {
		success = TRUE;
		g_task_return_pointer (task, info,
		                       (GDestroyNotify) tracker_extract_info_unref);
	} else {
		if (error) {
			g_task_return_error (task, error);
		} else {
			g_task_return_new_error (task,
			                         tracker_extract_error_quark (),
			                         TRACKER_EXTRACT_ERROR_NO_EXTRACTOR,
			                         "Could not get any metadata for uri:'%s' and mime:'%s'",
			                         data->file, data->mimetype);
		}
	}

#ifdef G_ENABLE_DEBUG
	if (TRACKER_DEBUG_CHECK (STATISTICS)) {
		if (data->module) {
			StatisticsData *stats_data;

			stats_data = g_hash_table_lookup (extract->statistics_data,
			                                  data->module);
			g_timer_stop (stats_data->elapsed);

			stats_data->extracted_count++;

			if (!success) {
				stats_data->failed_count++;
			}
		} else {
			extract->unhandled_count++;
		}
	}
#endif

	return FALSE;
}

static gboolean
handle_task_in_thread (gpointer user_data)
{
	g_autoptr (GTask) task = user_data;

	get_metadata (task);

	return G_SOURCE_REMOVE;
}


static gpointer
metadata_thread_func (GMainLoop *loop)
{
	GMainContext *context;

	context = g_main_loop_get_context (loop);
	g_main_context_push_thread_default (context);

	g_main_loop_run (loop);

	g_main_context_pop_thread_default (context);
	g_main_loop_unref (loop);

	return NULL;
}

void
tracker_extract_file (TrackerExtract      *extract,
                      const gchar         *file,
                      const gchar         *content_id,
                      const gchar         *mimetype,
                      GCancellable        *cancellable,
                      GAsyncReadyCallback  cb,
                      gpointer             user_data)
{
	g_autoptr (GTask) task = NULL;
	g_autoptr (GError) error = NULL;
	TrackerExtractTaskData *data;
	const char *graph;

	g_return_if_fail (TRACKER_IS_EXTRACT (extract));
	g_return_if_fail (file != NULL);
	g_return_if_fail (cb != NULL);

	if (!mimetype) {
		g_task_report_new_error (extract, cb, user_data, NULL,
		                         TRACKER_EXTRACT_ERROR,
		                         TRACKER_EXTRACT_ERROR_NO_MIMETYPE,
		                         "No mimetype for '%s'",
		                         file);
		return;
	}

	graph = tracker_extract_module_manager_get_graph (mimetype);

	if (!graph) {
		g_task_report_new_error (extract, cb, user_data, NULL,
		                         TRACKER_EXTRACT_ERROR,
		                         TRACKER_EXTRACT_ERROR_NO_EXTRACTOR,
		                         "Unknown target graph for uri:'%s' and mime:'%s'",
		                         file, mimetype);
		return;
	}

	task = g_task_new (extract, cancellable, cb, user_data);
	data = extract_task_data_new (extract, file, content_id, mimetype, graph);
	g_task_set_task_data (task, data, (GDestroyNotify) extract_task_data_free);

#ifdef G_ENABLE_DEBUG
	if (TRACKER_DEBUG_CHECK (STATISTICS)) {
		g_timer_continue (extract->total_elapsed);
	}
#endif

	if (!extract->task_thread) {
		extract->thread_context = g_main_context_new ();
		extract->thread_loop = g_main_loop_new (extract->thread_context, FALSE);

		extract->task_thread =
			g_thread_try_new ("single",
					  (GThreadFunc) metadata_thread_func,
					  g_main_loop_ref (extract->thread_loop),
					  &error);
		if (!extract->task_thread) {
			g_task_return_error (task, error);
			return;
		}

	}

	g_main_context_invoke (extract->thread_context,
	                       handle_task_in_thread,
	                       g_steal_pointer (&task));
}

TrackerExtractInfo *
tracker_extract_file_sync (TrackerExtract  *object,
                           const gchar     *uri,
                           const gchar     *content_id,
                           const gchar     *mimetype,
                           GError         **error)
{
	g_autoptr (TrackerExtractTaskData) task = NULL;
	TrackerExtractInfo *info;
	const char *graph = NULL;

	g_return_val_if_fail (uri != NULL, NULL);
	g_return_val_if_fail (content_id != NULL, NULL);

	if (mimetype)
		graph = tracker_extract_module_manager_get_graph (mimetype);

	if (!graph) {
		g_set_error (error,
		             TRACKER_EXTRACT_ERROR,
		             TRACKER_EXTRACT_ERROR_NO_EXTRACTOR,
		             "Unknown target graph for uri:'%s' and mime:'%s'",
		             uri, mimetype);
		return NULL;
	}

	task = extract_task_data_new (object, uri, content_id, mimetype, graph);

	if (!get_file_metadata (task, &info, error))
		return NULL;

	return info;
}

TrackerExtractInfo *
tracker_extract_file_finish (TrackerExtract  *extract,
                             GAsyncResult    *res,
                             GError         **error)
{
	g_return_val_if_fail (TRACKER_IS_EXTRACT (extract), NULL);
	g_return_val_if_fail (G_IS_ASYNC_RESULT (res), NULL);
	g_return_val_if_fail (!error || !*error, NULL);

#ifdef G_ENABLE_DEBUG
	if (TRACKER_DEBUG_CHECK (STATISTICS)) {
		g_timer_stop (extract->total_elapsed);
	}
#endif

	return g_task_propagate_pointer (G_TASK (res), error);
}

void
tracker_extract_set_max_text (TrackerExtract *extract,
                              gint            max_text)
{
	extract->max_text = max_text;
}
