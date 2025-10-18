/*
 * Copyright (C) 2009, Nokia <ivan.frade@nokia.com>
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

#include <tracker-common.h>

#include "tracker-miner-fs.h"
#include "tracker-monitor.h"
#include "tracker-utils.h"
#include "tracker-priority-queue.h"
#include "tracker-task-pool.h"
#include "tracker-sparql-buffer.h"
#include "tracker-file-notifier.h"
#include "tracker-lru.h"

#define BUFFER_POOL_LIMIT 800
#define DEFAULT_URN_LRU_SIZE 100

#define BIG_QUEUE_THRESHOLD 1000

/* Put tasks processing at a lower priority so other events
 * (timeouts, monitor events, etc...) are guaranteed to be
 * dispatched promptly.
 */
#define TRACKER_TASK_PRIORITY G_PRIORITY_DEFAULT_IDLE + 10

#define MAX_SIMULTANEOUS_ITEMS 64

#define TRACKER_CRAWLER_MAX_TIMEOUT_INTERVAL 1000

/**
 * SECTION:tracker-miner-fs
 * @short_description: Abstract base class for filesystem miners
 * @include: libtracker-miner/tracker-miner.h
 *
 * #TrackerMinerFS is an abstract base class for miners that collect data
 * from a filesystem where parent/child relationships need to be
 * inserted into the database correctly with queue management.
 *
 * All the filesystem crawling and monitoring is abstracted away,
 * leaving to implementations the decisions of what directories/files
 * should it process, and the actual data extraction.
 **/

typedef struct {
	guint16 type;
	guint attributes_update : 1;
	guint is_dir : 1;
	GFile *file;
	GFile *dest_file;
	GFileInfo *info;
	GList *queue_node;
} QueueEvent;

typedef struct {
	GFile *file;
	gchar *urn;
	gint priority;
	GCancellable *cancellable;
	TrackerMiner *miner;
	TrackerTask *task;
} UpdateProcessingTaskContext;

typedef struct _TrackerMinerFSPrivate TrackerMinerFSPrivate;

struct _TrackerMinerFSPrivate {
	TrackerPriorityQueue *items;
	GHashTable *items_by_file;

	guint item_queues_handler_id;

	TrackerMonitor *monitor;
	TrackerIndexingTree *indexing_tree;
	TrackerFileNotifier *file_notifier;

	/* Sparql insertion tasks */
	TrackerSparqlBuffer *sparql_buffer;

	/* Folder URN cache */
	TrackerLRU *urn_lru;

	/* Properties */
	gdouble throttle;

	/* Status */
	GTimer *timer;
	GTimer *extraction_timer;

	guint is_paused : 1;        /* TRUE if miner is paused */
	guint flushing : 1;         /* TRUE if flushing SPARQL */

	guint timer_stopped : 1;    /* TRUE if main timer is stopped */
	guint extraction_timer_stopped : 1; /* TRUE if the extraction
	                                     * timer is stopped */

	guint status_idle_id;
};

typedef enum {
	QUEUE_ACTION_NONE           = 0,
	QUEUE_ACTION_DELETE_FIRST   = 1 << 0,
	QUEUE_ACTION_DELETE_SECOND  = 1 << 1,
} QueueCoalesceAction;

typedef enum {
	TRACKER_MINER_FS_EVENT_CREATED,
	TRACKER_MINER_FS_EVENT_UPDATED,
	TRACKER_MINER_FS_EVENT_DELETED,
	TRACKER_MINER_FS_EVENT_MOVED,
	TRACKER_MINER_FS_EVENT_FINISH_DIRECTORY,
} TrackerMinerFSEventType;

enum {
	FINISHED,
	CORRUPT,
	NO_SPACE,
	LAST_SIGNAL
};

static guint signals[LAST_SIGNAL] = { 0, };

enum {
	PROP_0,
	PROP_THROTTLE,
	PROP_INDEXING_TREE,
	PROP_MONITOR,
	N_PROPS,
};

static GParamSpec *props[N_PROPS] = { 0, };

static void           fs_finalize                         (GObject              *object);
static void           fs_constructed                      (GObject              *object);
static void           fs_set_property                     (GObject              *object,
                                                           guint                 prop_id,
                                                           const GValue         *value,
                                                           GParamSpec           *pspec);
static void           fs_get_property                     (GObject              *object,
                                                           guint                 prop_id,
                                                           GValue               *value,
                                                           GParamSpec           *pspec);

static void           miner_started                       (TrackerMiner         *miner);
static void           miner_stopped                       (TrackerMiner         *miner);
static void           miner_paused                        (TrackerMiner         *miner);
static void           miner_resumed                       (TrackerMiner         *miner);

static void           indexing_tree_directory_removed     (TrackerIndexingTree  *indexing_tree,
                                                           GFile                *directory,
                                                           gpointer              user_data);
static void           file_notifier_file_created          (TrackerFileNotifier  *notifier,
                                                           GFile                *file,
                                                           GFileInfo            *info,
                                                           gpointer              user_data);
static void           file_notifier_file_deleted          (TrackerFileNotifier  *notifier,
                                                           GFile                *file,
                                                           gboolean              is_dir,
                                                           gpointer              user_data);
static void           file_notifier_file_updated          (TrackerFileNotifier  *notifier,
                                                           GFile                *file,
                                                           GFileInfo            *info,
                                                           gboolean              attributes_only,
                                                           gpointer              user_data);
static void           file_notifier_file_moved            (TrackerFileNotifier  *notifier,
                                                           GFile                *source,
                                                           GFile                *dest,
                                                           gboolean              is_dir,
                                                           gpointer              user_data);

static void           file_notifier_directory_finished (TrackerFileNotifier *notifier,
                                                        GFile               *directory,
                                                        gpointer             user_data);

static void           file_notifier_finished              (TrackerFileNotifier *notifier,
                                                           gpointer             user_data);

static void           queue_handler_maybe_set_up          (TrackerMinerFS       *fs);

static void           task_pool_limit_reached_notify_cb       (GObject        *object,
                                                               GParamSpec     *pspec,
                                                               gpointer        user_data);

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (TrackerMinerFS, tracker_miner_fs, TRACKER_TYPE_MINER)

#define EVENT_QUEUE_LOG_PREFIX "[Event Queue] "

G_GNUC_UNUSED static void
debug_print_event (QueueEvent *event)
{
	const gchar *event_type_name[] = { "CREATED", "UPDATED", "DELETED", "MOVED", "FINISH_DIRECTORY" };
	g_autofree char *uri1 = NULL, *uri2 = NULL;

	uri1 = g_file_get_uri (event->file);
	if (event->dest_file)
		uri2 = g_file_get_uri (event->dest_file);

	g_message ("%s New %s event: %s%s%s%s",
	            EVENT_QUEUE_LOG_PREFIX,
	            event_type_name[event->type],
	            event->attributes_update ? "(attributes only) " : "",
	            uri1,
	            uri2 ? "->" : "",
	            uri2 ? uri2 : "");
}

static void
tracker_miner_fs_class_init (TrackerMinerFSClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS (klass);
	TrackerMinerClass *miner_class = TRACKER_MINER_CLASS (klass);

	object_class->finalize = fs_finalize;
	object_class->constructed = fs_constructed;
	object_class->set_property = fs_set_property;
	object_class->get_property = fs_get_property;

	miner_class->started = miner_started;
	miner_class->stopped = miner_stopped;
	miner_class->paused  = miner_paused;
	miner_class->resumed = miner_resumed;

	props[PROP_THROTTLE] =
		g_param_spec_double ("throttle", NULL, NULL,
		                     0, 1, 0,
		                     G_PARAM_READWRITE |
		                     G_PARAM_STATIC_STRINGS);
	props[PROP_INDEXING_TREE] =
		g_param_spec_object ("indexing-tree", NULL, NULL,
		                     TRACKER_TYPE_INDEXING_TREE,
		                     G_PARAM_READWRITE |
		                     G_PARAM_CONSTRUCT_ONLY |
		                     G_PARAM_STATIC_STRINGS);
	props[PROP_MONITOR] =
		g_param_spec_object ("monitor", NULL, NULL,
		                     TRACKER_TYPE_MONITOR,
		                     G_PARAM_WRITABLE |
		                     G_PARAM_CONSTRUCT_ONLY |
		                     G_PARAM_STATIC_STRINGS);

	g_object_class_install_properties (object_class, N_PROPS, props);

	signals[FINISHED] =
		g_signal_new ("finished",
		              G_TYPE_FROM_CLASS (object_class),
		              G_SIGNAL_RUN_LAST,
		              G_STRUCT_OFFSET (TrackerMinerFSClass, finished),
		              NULL, NULL, NULL,
		              G_TYPE_NONE, 0);

	signals[CORRUPT] =
		g_signal_new ("corrupt",
		              G_TYPE_FROM_CLASS (object_class),
		              G_SIGNAL_RUN_LAST, 0,
		              NULL, NULL,
		              g_cclosure_marshal_VOID__VOID,
		              G_TYPE_NONE, 0);
	signals[NO_SPACE] =
		g_signal_new ("no-space",
		              G_TYPE_FROM_CLASS (object_class),
		              G_SIGNAL_RUN_LAST, 0,
		              NULL, NULL,
		              g_cclosure_marshal_VOID__VOID,
		              G_TYPE_NONE, 0);
}

static void
tracker_miner_fs_init (TrackerMinerFS *object)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (object);

	priv->timer = g_timer_new ();
	priv->extraction_timer = g_timer_new ();

	g_timer_stop (priv->timer);
	g_timer_stop (priv->extraction_timer);

	priv->timer_stopped = TRUE;
	priv->extraction_timer_stopped = TRUE;

	priv->items = tracker_priority_queue_new ();
	priv->items_by_file = g_hash_table_new_full (g_file_hash,
	                                             (GEqualFunc) g_file_equal,
	                                             g_object_unref, NULL);

	priv->urn_lru = tracker_lru_new (DEFAULT_URN_LRU_SIZE,
	                                 g_file_hash,
	                                 (GEqualFunc) g_file_equal,
	                                 g_object_unref,
	                                 g_free);
}

static QueueEvent *
queue_event_new (TrackerMinerFSEventType  type,
                 GFile                   *file,
                 GFileInfo               *info)
{
	QueueEvent *event;

	g_assert (type != TRACKER_MINER_FS_EVENT_MOVED);

	event = g_new0 (QueueEvent, 1);
	event->type = type;
	g_set_object (&event->file, file);
	g_set_object (&event->info, info);

	return event;
}

static QueueEvent *
queue_event_moved_new (GFile    *source,
                       GFile    *dest,
                       gboolean  is_dir)
{
	QueueEvent *event;

	event = g_new0 (QueueEvent, 1);
	event->type = TRACKER_MINER_FS_EVENT_MOVED;
	event->is_dir = !!is_dir;
	g_set_object (&event->dest_file, dest);
	g_set_object (&event->file, source);

	return event;
}

static void
queue_event_free (QueueEvent *event)
{
	g_clear_object (&event->dest_file);
	g_clear_object (&event->file);
	g_clear_object (&event->info);
	g_free (event);
}

static QueueCoalesceAction
queue_event_coalesce (const QueueEvent  *first,
		      const QueueEvent  *second,
		      QueueEvent       **replacement)
{
	if (!g_file_equal (first->file, second->file))
		return QUEUE_ACTION_NONE;

	*replacement = NULL;

	if (first->type == TRACKER_MINER_FS_EVENT_CREATED) {
		if (second->type == TRACKER_MINER_FS_EVENT_CREATED ||
		    (second->type == TRACKER_MINER_FS_EVENT_UPDATED &&
		     !second->attributes_update)) {
			return QUEUE_ACTION_DELETE_FIRST;
		} else if (second->type == TRACKER_MINER_FS_EVENT_MOVED) {
			*replacement = queue_event_new (TRACKER_MINER_FS_EVENT_CREATED,
			                                second->dest_file,
			                                NULL);
			return (QUEUE_ACTION_DELETE_FIRST |
				QUEUE_ACTION_DELETE_SECOND);
		} else if (second->type == TRACKER_MINER_FS_EVENT_DELETED) {
			/* We can't be sure that "create" is replacing a file
			 * here. Preserve the second event just in case.
			 */
			return QUEUE_ACTION_DELETE_FIRST;
		}
	} else if (first->type == TRACKER_MINER_FS_EVENT_UPDATED) {
		if (second->type == TRACKER_MINER_FS_EVENT_UPDATED) {
			if (first->attributes_update && !second->attributes_update)
				return QUEUE_ACTION_DELETE_FIRST;
			else
				return QUEUE_ACTION_DELETE_SECOND;
		} else if (second->type == TRACKER_MINER_FS_EVENT_DELETED) {
			return QUEUE_ACTION_DELETE_FIRST;
		}
	} else if (first->type == TRACKER_MINER_FS_EVENT_MOVED) {
		if (second->type == TRACKER_MINER_FS_EVENT_MOVED) {
			if (first->file != second->dest_file) {
				*replacement = queue_event_moved_new (first->file,
				                                      second->dest_file,
				                                      first->is_dir);
			}

			return (QUEUE_ACTION_DELETE_FIRST |
				QUEUE_ACTION_DELETE_SECOND);
		} else if (second->type == TRACKER_MINER_FS_EVENT_DELETED) {
			*replacement = queue_event_new (TRACKER_MINER_FS_EVENT_DELETED,
			                                first->file,
			                                NULL);
			return (QUEUE_ACTION_DELETE_FIRST |
				QUEUE_ACTION_DELETE_SECOND);
		}
	} else if (first->type == TRACKER_MINER_FS_EVENT_DELETED &&
		   second->type == TRACKER_MINER_FS_EVENT_DELETED) {
		return QUEUE_ACTION_DELETE_SECOND;
	}

	return QUEUE_ACTION_NONE;
}

static gboolean
queue_event_is_equal_or_descendant (QueueEvent *event,
				    GFile      *prefix)
{
	return (g_file_equal (event->file, prefix) ||
		g_file_has_prefix (event->file, prefix));
}

static void
fs_finalize (GObject *object)
{
	TrackerMinerFS *fs = TRACKER_MINER_FS (object);
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	g_timer_destroy (priv->timer);
	g_timer_destroy (priv->extraction_timer);

	g_clear_handle_id (&priv->status_idle_id, g_source_remove);

	g_clear_pointer (&priv->urn_lru, tracker_lru_unref);
	g_clear_handle_id (&priv->item_queues_handler_id, g_source_remove);

	if (priv->file_notifier)
		tracker_file_notifier_stop (priv->file_notifier);

	g_clear_object (&priv->sparql_buffer);
	g_hash_table_unref (priv->items_by_file);

	tracker_priority_queue_foreach (priv->items,
					(GFunc) queue_event_free,
					NULL);
	tracker_priority_queue_unref (priv->items);

	g_clear_object (&priv->indexing_tree);
	g_clear_object (&priv->file_notifier);
	g_clear_object (&priv->monitor);

	G_OBJECT_CLASS (tracker_miner_fs_parent_class)->finalize (object);
}

static void
fs_constructed (GObject *object)
{
	TrackerMinerFS *fs = TRACKER_MINER_FS (object);
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	/* NOTE: We have to do this in this order because initables
	 * are called _AFTER_ constructed and for subclasses that are
	 * not initables we don't have any other way than to chain
	 * constructed and root/indexing tree must exist at that
	 * point.
	 *
	 * If priv->indexing_tree is NULL after this function, the
	 * initiable functions will fail and this class will not be
	 * created anyway.
	 */
	G_OBJECT_CLASS (tracker_miner_fs_parent_class)->constructed (object);

	g_signal_connect (priv->indexing_tree, "directory-removed",
	                  G_CALLBACK (indexing_tree_directory_removed),
	                  object);

	priv->sparql_buffer = tracker_sparql_buffer_new (tracker_miner_get_connection (TRACKER_MINER (object)),
	                                                 BUFFER_POOL_LIMIT);
	g_signal_connect (priv->sparql_buffer, "notify::limit-reached",
	                  G_CALLBACK (task_pool_limit_reached_notify_cb),
	                  object);

	/* Create the file notifier */
	priv->file_notifier = tracker_file_notifier_new (priv->indexing_tree,
	                                                 tracker_miner_get_connection (TRACKER_MINER (object)),
	                                                 priv->monitor);

	g_signal_connect (priv->file_notifier, "file-created",
	                  G_CALLBACK (file_notifier_file_created),
	                  object);
	g_signal_connect (priv->file_notifier, "file-updated",
	                  G_CALLBACK (file_notifier_file_updated),
	                  object);
	g_signal_connect (priv->file_notifier, "file-deleted",
	                  G_CALLBACK (file_notifier_file_deleted),
	                  object);
	g_signal_connect (priv->file_notifier, "file-moved",
	                  G_CALLBACK (file_notifier_file_moved),
	                  object);
	g_signal_connect (priv->file_notifier, "directory-finished",
	                  G_CALLBACK (file_notifier_directory_finished),
	                  object);
	g_signal_connect (priv->file_notifier, "finished",
	                  G_CALLBACK (file_notifier_finished),
	                  object);

	g_object_set (object,
	              "progress", 0.0,
	              "status", "Initializing",
	              "remaining-time", -1,
	              NULL);
}

static void
fs_set_property (GObject      *object,
                 guint         prop_id,
                 const GValue *value,
                 GParamSpec   *pspec)
{
	TrackerMinerFS *fs = TRACKER_MINER_FS (object);
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	switch (prop_id) {
	case PROP_THROTTLE:
		tracker_miner_fs_set_throttle (TRACKER_MINER_FS (object),
		                               g_value_get_double (value));
		break;
	case PROP_INDEXING_TREE:
		priv->indexing_tree = g_value_dup_object (value);
		break;
	case PROP_MONITOR:
		priv->monitor = g_value_dup_object (value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

static void
fs_get_property (GObject    *object,
                 guint       prop_id,
                 GValue     *value,
                 GParamSpec *pspec)
{
	TrackerMinerFS *fs = TRACKER_MINER_FS (object);
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	switch (prop_id) {
	case PROP_THROTTLE:
		g_value_set_double (value, priv->throttle);
		break;
	case PROP_INDEXING_TREE:
		g_value_set_object (value, priv->indexing_tree);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

static void
task_pool_limit_reached_notify_cb (GObject    *object,
				   GParamSpec *pspec,
				   gpointer    user_data)
{
	if (!tracker_task_pool_limit_reached (TRACKER_TASK_POOL (object)))
		queue_handler_maybe_set_up (user_data);
}

static void
miner_started (TrackerMiner *miner)
{
	TrackerMinerFS *fs = TRACKER_MINER_FS (miner);
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	if (priv->timer_stopped) {
		g_timer_start (priv->timer);
		priv->timer_stopped = FALSE;
	}

	tracker_file_notifier_start (priv->file_notifier);
}

static void
miner_stopped (TrackerMiner *miner)
{
}

static void
miner_paused (TrackerMiner *miner)
{
	TrackerMinerFS *fs = TRACKER_MINER_FS (miner);
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	priv->is_paused = TRUE;

	tracker_file_notifier_stop (priv->file_notifier);

	if (priv->item_queues_handler_id) {
		g_source_remove (priv->item_queues_handler_id);
		priv->item_queues_handler_id = 0;
	}
}

static void
miner_resumed (TrackerMiner *miner)
{
	TrackerMinerFS *fs = TRACKER_MINER_FS (miner);
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	priv->is_paused = FALSE;

	tracker_file_notifier_start (priv->file_notifier);

	/* Only set up queue handler if we have items waiting to be
	 * processed.
	 */
	if (tracker_miner_fs_has_items_to_process (fs))
		queue_handler_maybe_set_up (fs);
}

static void
process_stop (TrackerMinerFS *fs)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	g_timer_stop (priv->timer);
	g_timer_stop (priv->extraction_timer);

	priv->timer_stopped = TRUE;
	priv->extraction_timer_stopped = TRUE;

	g_clear_handle_id (&priv->status_idle_id, g_source_remove);

	g_signal_emit (fs, signals[FINISHED], 0);
}

static void
check_notifier_high_water (TrackerMinerFS *fs)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	gboolean high_water;

	/* If there is more than worth 2 batches left processing, we can tell
	 * the notifier to stop a bit.
	 */
	high_water = (tracker_priority_queue_get_length (priv->items) >
	              2 * BUFFER_POOL_LIMIT);
	tracker_file_notifier_set_high_water (priv->file_notifier, high_water);
}

static void
sparql_buffer_flush_cb (GObject      *object,
                        GAsyncResult *result,
                        gpointer      user_data)
{
	TrackerMinerFS *fs = user_data;
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	GPtrArray *tasks;
	GError *error = NULL;
	TrackerTask *task;
	GFile *task_file;
	guint i;

	tasks = tracker_sparql_buffer_flush_finish (TRACKER_SPARQL_BUFFER (object),
	                                            result, &error);

	priv->flushing = FALSE;

	if (error) {
		g_warning ("Could not execute sparql: %s", error->message);

		if (g_error_matches (error, TRACKER_SPARQL_ERROR, TRACKER_SPARQL_ERROR_CORRUPT) ||
		    g_error_matches (error, TRACKER_SPARQL_ERROR, TRACKER_SPARQL_ERROR_CONSTRAINT)) {
			g_signal_emit (fs, signals[CORRUPT], 0);
			return;
		} else if (g_error_matches (error, TRACKER_SPARQL_ERROR,
		                            TRACKER_SPARQL_ERROR_NO_SPACE)) {
			g_signal_emit (fs, signals[NO_SPACE], 0);
			return;
		}
	}

	for (i = 0; i < tasks->len; i++) {
		task = g_ptr_array_index (tasks, i);
		task_file = tracker_task_get_file (task);

		if (error) {
			g_autofree char *sparql = NULL;

			sparql = tracker_sparql_task_get_sparql (task);
			tracker_error_report (task_file, error->message, sparql);
		} else {
			tracker_error_report_delete (task_file);
		}
	}

	if (tracker_task_pool_limit_reached (TRACKER_TASK_POOL (object))) {
		if (tracker_sparql_buffer_flush (TRACKER_SPARQL_BUFFER (object),
						 "SPARQL buffer again full after flush",
						 sparql_buffer_flush_cb,
						 fs))
			priv->flushing = TRUE;
	}

	queue_handler_maybe_set_up (fs);

	g_ptr_array_unref (tasks);
	g_clear_error (&error);
}

static void
item_add_or_update (TrackerMinerFS *fs,
                    GFile          *file,
                    GFileInfo      *file_info,
                    gboolean        attributes_update,
                    gboolean        create)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	g_autoptr (GFileInfo) info = NULL;
	g_autofree char *uri = NULL;

	if (file_info) {
		g_set_object (&info, file_info);
	} else {
		info = g_file_query_info (file,
		                          INDEXER_FILE_ATTRIBUTES,
		                          G_FILE_QUERY_INFO_NOFOLLOW_SYMLINKS,
		                          NULL, NULL);
		if (!info)
			return;
	}

	if (!create) {
		tracker_lru_remove (priv->urn_lru, file);
	}

	uri = g_file_get_uri (file);

	if (!attributes_update) {
		TRACKER_NOTE (MINER_FS_EVENTS, g_message ("Processing file '%s'...", uri));
		TRACKER_MINER_FS_GET_CLASS (fs)->process_file (fs, file, info,
		                                               priv->sparql_buffer,
		                                               create);
	} else {
		TRACKER_NOTE (MINER_FS_EVENTS, g_message ("Processing attributes in file '%s'...", uri));
		TRACKER_MINER_FS_GET_CLASS (fs)->process_file_attributes (fs, file, info,
		                                                          priv->sparql_buffer);
	}
}

static void
item_remove (TrackerMinerFS *fs,
             GFile          *file,
             gboolean        is_dir)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	g_autofree char *uri = NULL;

	uri = g_file_get_uri (file);

	TRACKER_NOTE (MINER_FS_EVENTS,
	              g_message ("Removing item: '%s' (Deleted from filesystem or no longer monitored)", uri));

	if (is_dir) {
		tracker_lru_remove_foreach (priv->urn_lru,
		                            (GEqualFunc) g_file_has_prefix,
		                            file);
	}

	tracker_lru_remove (priv->urn_lru, file);

	TRACKER_MINER_FS_GET_CLASS (fs)->remove_file (fs, file,
	                                              priv->sparql_buffer,
	                                              is_dir);
}

static void
item_move (TrackerMinerFS *fs,
           GFile          *dest_file,
           GFile          *source_file,
           gboolean        is_dir)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	g_autofree char *uri = NULL, *source_uri = NULL;
	TrackerDirectoryFlags source_flags, flags;
	gboolean source_recursive, dest_recursive;

	uri = g_file_get_uri (dest_file);
	source_uri = g_file_get_uri (source_file);

	TRACKER_NOTE (MINER_FS_EVENTS,
	              g_message ("Moving item from '%s' to '%s'",
	                         source_uri, uri));

	tracker_indexing_tree_get_root (priv->indexing_tree, source_file, NULL, &source_flags);
	source_recursive = (source_flags & TRACKER_DIRECTORY_FLAG_RECURSE) != 0;
	tracker_indexing_tree_get_root (priv->indexing_tree, dest_file, NULL, &flags);
	dest_recursive = (flags & TRACKER_DIRECTORY_FLAG_RECURSE) != 0;

	if (is_dir) {
		tracker_lru_remove_foreach (priv->urn_lru,
		                            (GEqualFunc) g_file_has_prefix,
		                            source_file);
	}

	tracker_lru_remove (priv->urn_lru, source_file);
	tracker_lru_remove (priv->urn_lru, dest_file);

	/* If the original location is recursive, but the destination location
	 * is not, remove all children.
	 */
	if (is_dir && source_recursive && !dest_recursive) {
		TRACKER_NOTE (MINER_FS_EVENTS,
		              g_message ("Removing children for item: '%s' (No longer monitored)", source_uri));

		TRACKER_MINER_FS_GET_CLASS (fs)->remove_children (fs, source_file,
		                                                  priv->sparql_buffer);
	}

	TRACKER_MINER_FS_GET_CLASS (fs)->move_file (fs, dest_file, source_file,
	                                            priv->sparql_buffer,
	                                            (is_dir && source_recursive &&
	                                             dest_recursive));
}

static void
item_finish_directory (TrackerMinerFS *fs,
                       GFile          *file)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	TRACKER_MINER_FS_GET_CLASS (fs)->finish_directory (fs, file, priv->sparql_buffer);
}

static gboolean
maybe_remove_file_event_node (TrackerMinerFS *fs,
                              QueueEvent     *event)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	QueueEvent *item_event;

	item_event = g_hash_table_lookup (priv->items_by_file, event->file);

	if (item_event == event) {
		g_hash_table_remove (priv->items_by_file, event->file);
		return TRUE;
	}

	return FALSE;
}

static gboolean
remove_items_by_file_foreach (gpointer key,
                              gpointer value,
                              gpointer user_data)
{
	GFile *file = key;
	GFile *prefix = user_data;

	return (g_file_equal (file, prefix) ||
	        g_file_has_prefix (file, prefix));
}

static void
item_queue_get_next_file (TrackerMinerFS           *fs,
                          GFile                   **file,
                          GFile                   **source_file,
                          GFileInfo               **info,
                          TrackerMinerFSEventType  *type,
                          gboolean                 *attributes_update,
                          gboolean                 *is_dir)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	QueueEvent *event;

	*file = NULL;
	*source_file = NULL;

	event = tracker_priority_queue_pop (priv->items, NULL);

	if (event) {
		if (event->type == TRACKER_MINER_FS_EVENT_MOVED) {
			g_set_object (file, event->dest_file);
			g_set_object (source_file, event->file);
		} else {
			g_set_object (file, event->file);
		}

		*type = event->type;
		*attributes_update = event->attributes_update;
		*is_dir = event->is_dir;
		g_set_object (info, event->info);

		maybe_remove_file_event_node (fs, event);
		queue_event_free (event);
	}
}

static gboolean
miner_handle_next_item (TrackerMinerFS *fs)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	GFile *file = NULL;
	GFile *source_file = NULL;
	gboolean keep_processing = TRUE;
	gboolean attributes_update = FALSE;
	gboolean is_dir = FALSE;
	TrackerMinerFSEventType type;
	GFileInfo *info = NULL;

	item_queue_get_next_file (fs, &file, &source_file, &info, &type,
	                          &attributes_update, &is_dir);

	if (priv->timer_stopped) {
		g_timer_start (priv->timer);
		priv->timer_stopped = FALSE;
	}

	if (file == NULL && !priv->extraction_timer_stopped) {
		g_timer_stop (priv->extraction_timer);
		priv->extraction_timer_stopped = TRUE;
	} else if (file != NULL && priv->extraction_timer_stopped) {
		g_timer_continue (priv->extraction_timer);
		priv->extraction_timer_stopped = FALSE;
	}

	if (file == NULL) {
		if (!tracker_file_notifier_is_active (priv->file_notifier)) {
			if (!priv->flushing &&
			    tracker_task_pool_get_size (TRACKER_TASK_POOL (priv->sparql_buffer)) == 0) {
				process_stop (fs);
			} else {
				/* Flush any possible pending update here */
				if (tracker_sparql_buffer_flush (priv->sparql_buffer,
								 "Queue handlers NONE",
								 sparql_buffer_flush_cb,
								 fs))
					priv->flushing = TRUE;
			}
		}

		/* No more files left to process */
		return FALSE;
	}

	/* Handle queues */
	switch (type) {
	case TRACKER_MINER_FS_EVENT_MOVED:
		item_move (fs, file, source_file, is_dir);
		break;
	case TRACKER_MINER_FS_EVENT_DELETED:
		item_remove (fs, file, is_dir);
		break;
	case TRACKER_MINER_FS_EVENT_CREATED:
		item_add_or_update (fs, file, info, FALSE, TRUE);
		break;
	case TRACKER_MINER_FS_EVENT_UPDATED:
		item_add_or_update (fs, file, info, attributes_update, FALSE);
		break;
	case TRACKER_MINER_FS_EVENT_FINISH_DIRECTORY:
		item_finish_directory (fs, file);
		break;
	default:
		g_assert_not_reached ();
	}

	if (tracker_task_pool_limit_reached (TRACKER_TASK_POOL (priv->sparql_buffer))) {
		if (tracker_sparql_buffer_flush (priv->sparql_buffer,
						 "SPARQL buffer limit reached",
						 sparql_buffer_flush_cb,
						 fs)) {
			priv->flushing = TRUE;
		} else {
			/* If we cannot flush, wait for the pending operations
			 * to finish.
			 */
			keep_processing = FALSE;
		}
	}

	check_notifier_high_water (fs);

	g_clear_object (&file);
	g_clear_object (&source_file);
	g_clear_object (&info);

	return keep_processing;
}

static gboolean
item_queue_handlers_cb (gpointer user_data)
{
	TrackerMinerFS *fs = user_data;
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	gint i;

	for (i = 0; i < MAX_SIMULTANEOUS_ITEMS; i++) {
		if (!miner_handle_next_item (fs)) {
			priv->item_queues_handler_id = 0;
			return G_SOURCE_REMOVE;
		}
	}

	return G_SOURCE_CONTINUE;
}

static gboolean
update_status_cb (gpointer user_data)
{
	TrackerMinerFS *fs = user_data;
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	g_autofree char *status = NULL;
	guint files_found, files_updated, files_ignored, files_reindexed;
	TrackerFileNotifierStatus notifier_status;
	GFile *current_root;

	if (tracker_file_notifier_get_status (priv->file_notifier,
	                                      &notifier_status,
	                                      &current_root,
	                                      &files_found,
	                                      &files_updated,
	                                      &files_ignored,
	                                      &files_reindexed)) {
		g_autofree char *uri = NULL;
		GString *str;

		str = g_string_new (NULL);
		uri = g_file_get_uri (current_root);

		if (notifier_status == TRACKER_FILE_NOTIFIER_STATUS_INDEXING)
			g_string_append_printf (str, "Indexing '%s'. ", uri);
		else if (notifier_status == TRACKER_FILE_NOTIFIER_STATUS_CHECKING)
			g_string_append_printf (str, "Checking '%s'. ", uri);

		if (files_found > 0)
			g_string_append_printf (str, "Found: %d. ", files_found);
		if (files_updated > 0)
			g_string_append_printf (str, "Updated: %d. ", files_updated);
		if (files_reindexed > 0)
			g_string_append_printf (str, "Re-indexed: %d. ", files_reindexed);
		if (files_ignored > 0)
			g_string_append_printf (str, "Ignored: %d. ", files_ignored);

		status = g_string_free (str, FALSE);
	} else {
		guint elems_left;

		elems_left =
			tracker_priority_queue_get_length (priv->items) +
			tracker_task_pool_get_size (TRACKER_TASK_POOL (priv->sparql_buffer));

		if (elems_left > 0)
			status = g_strdup_printf ("Processing %d updates…", elems_left);
	}

	if (status != NULL) {
		g_object_set (fs,
		              "status", status,
		              "progress", 0.0,
		              "remaining-time", -1,
		              NULL);
	}

	return G_SOURCE_CONTINUE;
}

static void
queue_handler_set_up (TrackerMinerFS *fs)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	guint source_id;
	guint interval;

	g_assert (priv->item_queues_handler_id == 0);
	interval = TRACKER_CRAWLER_MAX_TIMEOUT_INTERVAL * priv->throttle;

	if (interval == 0) {
		source_id = g_idle_add_full (TRACKER_TASK_PRIORITY,
		                             item_queue_handlers_cb,
		                             fs, NULL);
	} else {
		source_id = g_timeout_add_full (TRACKER_TASK_PRIORITY, interval,
		                                item_queue_handlers_cb,
		                                fs, NULL);
	}

	priv->item_queues_handler_id = source_id;
}

static void
queue_handler_maybe_set_up (TrackerMinerFS *fs)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	TRACKER_NOTE (MINER_FS_EVENTS, g_message (EVENT_QUEUE_LOG_PREFIX "Setting up queue handlers..."));

	if (priv->item_queues_handler_id != 0) {
		TRACKER_NOTE (MINER_FS_EVENTS, g_message (EVENT_QUEUE_LOG_PREFIX "   cancelled: already one active"));
		return;
	}

	if (priv->is_paused) {
		TRACKER_NOTE (MINER_FS_EVENTS, g_message (EVENT_QUEUE_LOG_PREFIX "   cancelled: paused"));
		return;
	}

	/* Already processing max number of sparql updates */
	if (tracker_task_pool_limit_reached (TRACKER_TASK_POOL (priv->sparql_buffer))) {
		TRACKER_NOTE (MINER_FS_EVENTS,
		              g_message (EVENT_QUEUE_LOG_PREFIX "   cancelled: pool limit reached (sparql buffer: %u)",
		                         tracker_task_pool_get_limit (TRACKER_TASK_POOL (priv->sparql_buffer))));
		return;
	}

	if (priv->status_idle_id == 0) {
		priv->status_idle_id = g_timeout_add (250,
		                                      update_status_cb,
		                                      fs);
		update_status_cb (fs);
	}

	TRACKER_NOTE (MINER_FS_EVENTS, g_message (EVENT_QUEUE_LOG_PREFIX "   scheduled in idle"));
	queue_handler_set_up (fs);
}

static gint
miner_fs_get_queue_priority (TrackerMinerFS *fs,
                             GFile          *file)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	TrackerDirectoryFlags flags;

	tracker_indexing_tree_get_root (priv->indexing_tree,
	                                file, NULL, &flags);

	return (flags & TRACKER_DIRECTORY_FLAG_PRIORITY) ?
	        G_PRIORITY_HIGH : G_PRIORITY_DEFAULT;
}

static void
miner_fs_queue_event (TrackerMinerFS *fs,
		      QueueEvent     *event,
		      guint           priority)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	QueueEvent *old = NULL;

	old = g_hash_table_lookup (priv->items_by_file, event->file);

	if (old) {
		QueueCoalesceAction action;
		QueueEvent *replacement = NULL;

		action = queue_event_coalesce (old, event, &replacement);

		if (action & QUEUE_ACTION_DELETE_FIRST) {
			g_hash_table_remove (priv->items_by_file, old->file);
			tracker_priority_queue_remove_node (priv->items,
							    old->queue_node);
			queue_event_free (old);
		}

		if (action & QUEUE_ACTION_DELETE_SECOND) {
			queue_event_free (event);
			event = NULL;
		}

		if (replacement)
			event = replacement;
	}

	if (event) {
		if (event->is_dir &&
		    event->type == TRACKER_MINER_FS_EVENT_DELETED &&
		    g_hash_table_size (priv->items_by_file) < BIG_QUEUE_THRESHOLD) {
			/* Attempt to optimize by removing any children
			 * of this directory from being processed.
			 */
			g_hash_table_foreach_remove (priv->items_by_file,
			                             remove_items_by_file_foreach,
			                             event->file);
			tracker_priority_queue_foreach_remove (priv->items,
							       (GEqualFunc) queue_event_is_equal_or_descendant,
							       event->file,
							       (GDestroyNotify) queue_event_free);
		}

		TRACKER_NOTE (MINER_FS_EVENTS, debug_print_event (event));

		event->queue_node =
			tracker_priority_queue_add (priv->items, event, priority);

		if (event->type == TRACKER_MINER_FS_EVENT_MOVED) {
			if (event->is_dir) {
				g_hash_table_foreach_remove (priv->items_by_file,
				                             remove_items_by_file_foreach,
				                             event->dest_file);
			} else {
				g_hash_table_remove (priv->items_by_file, event->dest_file);
			}

			g_hash_table_remove (priv->items_by_file, event->file);
		} else {
			g_hash_table_replace (priv->items_by_file,
			                      g_object_ref (event->file),
			                      event);
		}

		queue_handler_maybe_set_up (fs);
		check_notifier_high_water (fs);
	}
}

static void
file_notifier_file_created (TrackerFileNotifier  *notifier,
                            GFile                *file,
                            GFileInfo            *info,
                            gpointer              user_data)
{
	TrackerMinerFS *fs = user_data;
	QueueEvent *event;

	event = queue_event_new (TRACKER_MINER_FS_EVENT_CREATED, file, info);
	miner_fs_queue_event (fs, event, miner_fs_get_queue_priority (fs, file));
}

static void
file_notifier_file_deleted (TrackerFileNotifier  *notifier,
                            GFile                *file,
                            gboolean              is_dir,
                            gpointer              user_data)
{
	TrackerMinerFS *fs = user_data;
	QueueEvent *event;

	event = queue_event_new (TRACKER_MINER_FS_EVENT_DELETED, file, NULL);
	event->is_dir = !!is_dir;
	miner_fs_queue_event (fs, event, miner_fs_get_queue_priority (fs, file));
}

static void
file_notifier_file_updated (TrackerFileNotifier  *notifier,
                            GFile                *file,
                            GFileInfo            *info,
                            gboolean              attributes_only,
                            gpointer              user_data)
{
	TrackerMinerFS *fs = user_data;
	QueueEvent *event;

	event = queue_event_new (TRACKER_MINER_FS_EVENT_UPDATED, file, info);
	event->attributes_update = attributes_only;
	miner_fs_queue_event (fs, event, miner_fs_get_queue_priority (fs, file));
}

static void
file_notifier_file_moved (TrackerFileNotifier *notifier,
                          GFile               *source,
                          GFile               *dest,
                          gboolean             is_dir,
                          gpointer             user_data)
{
	TrackerMinerFS *fs = user_data;
	QueueEvent *event;

	event = queue_event_moved_new (source, dest, is_dir);
	miner_fs_queue_event (fs, event, miner_fs_get_queue_priority (fs, source));
}

static void
file_notifier_directory_finished (TrackerFileNotifier *notifier,
                                  GFile               *directory,
                                  gpointer             user_data)
{
	TrackerMinerFS *fs = user_data;
	QueueEvent *event;

	event = queue_event_new (TRACKER_MINER_FS_EVENT_FINISH_DIRECTORY, directory, NULL);
	miner_fs_queue_event (fs, event, miner_fs_get_queue_priority (fs, directory));
}

static void
file_notifier_finished (TrackerFileNotifier *notifier,
                        gpointer             user_data)
{
	TrackerMinerFS *fs = user_data;

	queue_handler_maybe_set_up (fs);
}

static void
indexing_tree_directory_removed (TrackerIndexingTree *indexing_tree,
                                 GFile               *directory,
                                 gpointer             user_data)
{
	TrackerMinerFS *fs = user_data;
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	GTimer *timer = g_timer_new ();

	TRACKER_NOTE (MINER_FS_EVENTS, g_message ("  Cancelled processing pool tasks at %f\n", g_timer_elapsed (timer, NULL)));

	/* Remove anything contained in the removed directory
	 * from all relevant processing queues.
	 */
	g_hash_table_foreach_remove (priv->items_by_file,
	                             remove_items_by_file_foreach,
	                             directory);
	tracker_priority_queue_foreach_remove (priv->items,
					       (GEqualFunc) queue_event_is_equal_or_descendant,
					       directory,
					       (GDestroyNotify) queue_event_free);

	TRACKER_NOTE (MINER_FS_EVENTS, g_message ("  Removed files at %f\n", g_timer_elapsed (timer, NULL)));
	g_timer_destroy (timer);
}

/**
 * tracker_miner_fs_set_throttle:
 * @fs: a #TrackerMinerFS
 * @throttle: a double between 0.0 and 1.0
 *
 * Tells the filesystem miner to throttle its operations. A value of
 * 0.0 means no throttling at all, so the miner will perform
 * operations at full speed, 1.0 is the slowest value. With a value of
 * 1.0, the @fs is typically waiting one full second before handling
 * the next batch of queued items to be processed.
 *
 * Since: 0.8
 **/
void
tracker_miner_fs_set_throttle (TrackerMinerFS *fs,
                               gdouble         throttle)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	g_return_if_fail (TRACKER_IS_MINER_FS (fs));

	throttle = CLAMP (throttle, 0, 1);

	if (priv->throttle == throttle) {
		return;
	}

	priv->throttle = throttle;

	/* Update timeouts */
	if (priv->item_queues_handler_id != 0) {
		g_clear_handle_id (&priv->item_queues_handler_id, g_source_remove);
		queue_handler_set_up (fs);
	}
}

/**
 * tracker_miner_fs_get_throttle:
 * @fs: a #TrackerMinerFS
 *
 * Gets the current throttle value, see
 * tracker_miner_fs_set_throttle() for more details.
 *
 * Returns: a double representing a value between 0.0 and 1.0.
 *
 * Since: 0.8
 **/
gdouble
tracker_miner_fs_get_throttle (TrackerMinerFS *fs)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	g_return_val_if_fail (TRACKER_IS_MINER_FS (fs), 0);

	return priv->throttle;
}

const gchar *
tracker_miner_fs_get_identifier (TrackerMinerFS *fs,
				 GFile          *file)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);
	g_autoptr (GFileInfo) info = NULL;
	gchar *str;

	g_return_val_if_fail (TRACKER_IS_MINER_FS (fs), NULL);
	g_return_val_if_fail (G_IS_FILE (file), NULL);

	if (tracker_lru_find (priv->urn_lru, file, (gpointer*) &str))
		return str;

	info = g_file_query_info (file,
	                          G_FILE_ATTRIBUTE_STANDARD_TYPE ","
	                          G_FILE_ATTRIBUTE_STANDARD_IS_HIDDEN ","
	                          G_FILE_ATTRIBUTE_ID_FILESYSTEM ","
	                          G_FILE_ATTRIBUTE_UNIX_INODE,
	                          G_FILE_QUERY_INFO_NONE,
	                          NULL,
	                          NULL);
	if (!info)
		return NULL;

	if (!tracker_indexing_tree_file_is_indexable (priv->indexing_tree, file, info))
		return NULL;

	str = TRACKER_MINER_FS_GET_CLASS (fs)->get_content_identifier (fs, file, info);
	tracker_lru_add (priv->urn_lru, g_object_ref (file), str);

	return str;
}

/**
 * tracker_miner_fs_has_items_to_process:
 * @fs: a #TrackerMinerFS
 *
 * The @fs keeps many priority queus for content it is processing.
 * This function returns %TRUE if the sum of all (or any) priority
 * queues is more than 0. This includes items deleted, created,
 * updated, moved or being written back.
 *
 * Returns: %TRUE if there are items to process in the internal
 * queues, otherwise %FALSE.
 *
 * Since: 0.10
 **/
gboolean
tracker_miner_fs_has_items_to_process (TrackerMinerFS *fs)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	g_return_val_if_fail (TRACKER_IS_MINER_FS (fs), FALSE);

	if (tracker_file_notifier_is_active (priv->file_notifier) ||
	    !tracker_priority_queue_is_empty (priv->items)) {
		return TRUE;
	}

	return FALSE;
}

/**
 * tracker_miner_fs_get_indexing_tree:
 * @fs: a #TrackerMinerFS
 *
 * Returns the #TrackerIndexingTree which determines
 * what files/directories are indexed by @fs
 *
 * Returns: (transfer none): The #TrackerIndexingTree
 *          holding the indexing configuration
 **/
TrackerIndexingTree *
tracker_miner_fs_get_indexing_tree (TrackerMinerFS *fs)
{
	TrackerMinerFSPrivate *priv =
		tracker_miner_fs_get_instance_private (fs);

	g_return_val_if_fail (TRACKER_IS_MINER_FS (fs), NULL);

	return priv->indexing_tree;
}
