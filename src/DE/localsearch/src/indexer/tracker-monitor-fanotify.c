/*
 * Copyright (C) 2021, Red Hat Inc
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
 * Author: Carlos Garnacho <carlosg@gnome.org>
 */

#include "config-miners.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <gio/gio.h>
#include <sys/fanotify.h>
#include <sys/vfs.h>

#include <glib-unix.h>

#include "tracker-monitor-fanotify.h"
#include "tracker-monitor-private.h"

#include <tracker-common.h>

#define FANOTIFY_EVENTS (FAN_CREATE | FAN_MODIFY | FAN_CLOSE_WRITE | \
                         FAN_ATTRIB | \
                         FAN_DELETE | FAN_DELETE_SELF |	\
                         FAN_MOVED_TO | FAN_MOVED_FROM | FAN_MOVE_SELF | \
                         FAN_EVENT_ON_CHILD | FAN_ONDIR)

typedef enum {
	EVENT_NONE,
	EVENT_CREATE,
	EVENT_UPDATE,
	EVENT_ATTRIBUTES_UPDATE,
	EVENT_DELETE,
	EVENT_MOVE,
} EventType;

typedef struct {
	EventType type;
	TrackerMonitor *monitor;
	GFile *file;
	gboolean is_directory;
} MonitorEvent;

struct _TrackerMonitorFanotify {
	TrackerMonitor parent_instance;

	GHashTable *monitored_dirs;
	GHashTable *handles;
	GHashTable *cached_events;
	GSource *source;
	gboolean enabled;
	int fanotify_fd;

	ssize_t file_handle_payload;
	GFile *moved_file;
	guint limit;
	guint ignored;
};

/* Binary compatible with the last portions of fanotify_event_info_fid */
typedef struct {
	fsid_t fsid;
	struct file_handle handle;
} HandleData;

typedef struct {
	TrackerMonitorFanotify *monitor;
	GFile *file;
	GBytes *handle_bytes;
	/* This must be last in the struct */
	HandleData handle;
} MonitoredFile;

enum {
	ITEM_CREATED,
	ITEM_UPDATED,
	ITEM_ATTRIBUTE_UPDATED,
	ITEM_DELETED,
	ITEM_MOVED,
	LAST_SIGNAL
};

enum {
	PROP_0,
	PROP_ENABLED,
	PROP_LIMIT,
	PROP_COUNT,
	PROP_IGNORED,
};

static GInitableIface *initable_parent_iface = NULL;

static void tracker_monitor_fanotify_initable_iface_init (GInitableIface *iface);
static void monitored_file_free (MonitoredFile *data);

G_DEFINE_TYPE_WITH_CODE (TrackerMonitorFanotify, tracker_monitor_fanotify,
                         TRACKER_TYPE_MONITOR_GLIB,
                         G_IMPLEMENT_INTERFACE (G_TYPE_INITABLE,
                                                tracker_monitor_fanotify_initable_iface_init))

static inline const char *
event_type_to_string (EventType evtype)
{
	switch (evtype) {
	case EVENT_CREATE:
		return "CREATE";
	case EVENT_UPDATE:
		return "UPDATE";
	case EVENT_ATTRIBUTES_UPDATE:
		return "ATTRIBUTES_UPDATE";
	case EVENT_DELETE:
		return "DELETE";
	case EVENT_MOVE:
		return "MOVE";
	default:
		g_assert_not_reached ();
	}
}

static void
emit_event (TrackerMonitorFanotify *monitor,
            EventType               evtype,
            GFile                  *file,
            GFile                  *other_file,
            gboolean                is_directory)
{
	if (evtype == EVENT_MOVE) {
		TRACKER_NOTE (MONITORS,
		              g_message ("Received monitor event:%d (%s) for files '%s'->'%s'",
		                         evtype,
		                         event_type_to_string (evtype),
		                         g_file_peek_path (file),
		                         g_file_peek_path (other_file)));
		tracker_monitor_emit_moved (TRACKER_MONITOR (monitor),
		                            file, other_file, is_directory);
	} else {
		TRACKER_NOTE (MONITORS,
		              g_message ("Received monitor event:%d (%s) for %s:'%s'",
		                         evtype,
		                         event_type_to_string (evtype),
		                         is_directory ? "directory" : "file",
		                         g_file_peek_path (file)));
		switch (evtype) {
		case EVENT_CREATE:
			tracker_monitor_emit_created (TRACKER_MONITOR (monitor),
			                              file, is_directory);
			break;
		case EVENT_UPDATE:
			tracker_monitor_emit_updated (TRACKER_MONITOR (monitor),
			                              file, is_directory);
			break;
		case EVENT_ATTRIBUTES_UPDATE:
			tracker_monitor_emit_attributes_updated (TRACKER_MONITOR (monitor),
			                                         file, is_directory);
			break;
		case EVENT_DELETE:
			tracker_monitor_emit_deleted (TRACKER_MONITOR (monitor),
			                              file, is_directory);
			break;
		default:
			g_assert_not_reached ();
		}
	}
}

static void
flush_event (TrackerMonitorFanotify *monitor,
             GFile                  *file)
{
	MonitorEvent *event;

	event = g_hash_table_lookup (monitor->cached_events, file);
	if (!event)
		return;

	emit_event (monitor, event->type, event->file, NULL, event->is_directory);
	g_hash_table_remove (monitor->cached_events, file);
}

static void
forget_event (TrackerMonitorFanotify *monitor,
              GFile                  *file)
{
	g_hash_table_remove (monitor->cached_events, file);
}

static void
monitor_event_free (MonitorEvent *event)
{
	g_object_unref (event->file);
	g_slice_free (MonitorEvent, event);
}

static void
cache_event (TrackerMonitorFanotify *monitor,
             EventType               evtype,
             GFile                  *file,
             gboolean                is_directory)
{
	MonitorEvent *event, *prev_event;

	prev_event = g_hash_table_lookup (monitor->cached_events, file);

	if (prev_event) {
		/* Check whether the prior event is compatible */
		if (evtype == EVENT_UPDATE && prev_event->type == EVENT_CREATE)
			return;
		if (evtype == EVENT_UPDATE && prev_event->type == EVENT_UPDATE)
			return;
		if (evtype == EVENT_DELETE && prev_event->type == EVENT_DELETE)
			return;

		/* Otherwise flush the event */
		flush_event (monitor, file);
	}

	event = g_slice_new0 (MonitorEvent);
	event->type = evtype;
	event->file = g_object_ref (file);
	event->is_directory = is_directory;

	g_hash_table_insert (monitor->cached_events, event->file, event);
}

static void
handle_monitor_events (TrackerMonitorFanotify *monitor,
                       GFile                  *file,
                       uint32_t                mask)
{
	gboolean is_directory;

	is_directory = (mask & FAN_ONDIR) != 0;

	if (mask & FAN_CREATE) {
		if (is_directory) {
			emit_event (monitor, EVENT_CREATE, file, NULL, is_directory);
		} else {
			GFileType file_type;

			file_type = g_file_query_file_type (file,
							    G_FILE_QUERY_INFO_NOFOLLOW_SYMLINKS,
							    NULL);

			if (file_type == G_FILE_TYPE_SYMBOLIC_LINK ||
			    file_type == G_FILE_TYPE_SPECIAL) {
				/* We will not get FAN_CLOSE for these file types */
				emit_event (monitor, EVENT_CREATE, file, NULL, is_directory);
			} else if (file_type != G_FILE_TYPE_UNKNOWN) {
				cache_event (monitor, EVENT_CREATE, file, is_directory);
			}
		}
	}

	if (mask & FAN_MODIFY) {
		if (is_directory) {
			emit_event (monitor, EVENT_UPDATE, file, NULL, is_directory);
		} else {
			cache_event (monitor, EVENT_UPDATE, file, is_directory);
		}
	}

	if (mask & FAN_ATTRIB) {
		emit_event (monitor, EVENT_ATTRIBUTES_UPDATE,
		            file, NULL, is_directory);
	}

	if (mask & (FAN_DELETE | FAN_DELETE_SELF)) {
		cache_event (monitor, EVENT_DELETE, file, is_directory);
		if (mask & FAN_DELETE)
			flush_event (monitor, file);
	}

	if (mask & FAN_CLOSE_WRITE) {
		/* Flush the CREATE/UPDATE event here */
		flush_event (monitor, file);
	}

	if (mask & FAN_MOVED_FROM) {
		cache_event (monitor, EVENT_DELETE, file, is_directory);
		g_set_object (&monitor->moved_file, file);
	}

	if (mask & FAN_MOVED_TO) {
		GFile *source_file;

		source_file = monitor->moved_file;

		if (source_file == NULL) {
			emit_event (monitor, EVENT_CREATE, file, NULL, is_directory);
		} else {
			forget_event (monitor, source_file);
			emit_event (monitor, EVENT_MOVE, source_file, file, is_directory);
		}

		g_clear_object (&monitor->moved_file);
	}
}

static inline GBytes *
create_bytes_for_handle (HandleData *handle)
{
	return g_bytes_new_static (handle,
	                           sizeof (HandleData) +
	                           handle->handle.handle_bytes);
}

static void
flush_moved_file_event (TrackerMonitorFanotify *monitor)
{
	if (monitor->moved_file) {
		flush_event (monitor, monitor->moved_file);
		g_clear_object (&monitor->moved_file);
	}
}

static gboolean
fanotify_events_cb (int          fd,
                    GIOCondition condition,
                    gpointer     user_data)
{
	TrackerMonitorFanotify *monitor = user_data;
	struct fanotify_event_metadata buf[200], *event;
	ssize_t len;

	len = read (monitor->fanotify_fd, buf, sizeof (buf));

	event = buf;

	while (FAN_EVENT_OK (event, len)) {
		struct fanotify_event_info_fid *fid;
		HandleData *handle;
		MonitoredFile *data;
		const gchar *file_name;
		GBytes *fid_bytes;
		GFile *child;

		/* Check that run-time and compile-time structures match. */
		if (event->vers != FANOTIFY_METADATA_VERSION) {
			g_warning ("Fanotify ABI mismatch, monitoring is disabled");
			return G_SOURCE_REMOVE;
		}

		/* We expect data as FID, not as a file descriptor */
		if (event->fd != FAN_NOFD) {
			TRACKER_NOTE (MONITORS, g_message ("Received a file descriptor unexpectedly"));
			goto cont;
		}

		fid = (struct fanotify_event_info_fid *) (event + 1);

		/* Ensure that the event info is of the correct type. */
		if (fid->hdr.info_type != FAN_EVENT_INFO_TYPE_DFID_NAME)
			goto cont;

		/* fsid/handle portions are compatible with HandleData */
		handle = (HandleData *) &fid->fsid;
		fid_bytes = create_bytes_for_handle (handle);
		data = g_hash_table_lookup (monitor->handles, fid_bytes);
		g_bytes_unref (fid_bytes);

		if (!data) {
			/* We are receiving a notification on an unknown handle,
			 * should this ever happen on folders? In either case this is
			 * ignored, presumably will be fixed by events that
			 * are yet to be handled.
			 */
			goto cont;
		}

		/* File name comes after the file handle data */
		file_name = handle->handle.f_handle + handle->handle.handle_bytes;

		if (g_strcmp0 (file_name, ".") == 0)
			child = g_object_ref (data->file);
		else
			child = g_file_get_child (data->file, file_name);

		/* We have a pending MOVED_FROM event, now unpaired. Flush
		 * it as a DELETE event, since it's moving outside our
		 * inspected folders.
		 */
		if (monitor->moved_file && (event->mask & FAN_MOVED_TO) == 0)
			flush_moved_file_event (monitor);

		handle_monitor_events (monitor, child, event->mask);
		g_object_unref (child);

	cont:
		event = FAN_EVENT_NEXT (event, len);
	}

	flush_moved_file_event (monitor);

	return G_SOURCE_CONTINUE;
}

static gboolean
get_fanotify_limit (guint   *limit,
                    GError **error)
{
	GError *inner_error = NULL;
	gchar *contents = NULL;

	if (!g_file_get_contents ("/proc/sys/fs/fanotify/max_user_marks",
	                          &contents,
	                          NULL,
	                          &inner_error)) {
		g_propagate_prefixed_error (error, inner_error,
		                            "Couldn't get Fanotify marks limit:");
		return FALSE;
	}

	if (limit)
		*limit = atoi (contents);

	g_free (contents);
	return TRUE;
}

static gboolean
tracker_monitor_fanotify_initable_init (GInitable     *initable,
                                        GCancellable  *cancellable,
                                        GError       **error)
{
	TrackerMonitorFanotify *monitor = TRACKER_MONITOR_FANOTIFY (initable);
	guint limit;

	TRACKER_NOTE (MONITORS, g_message ("Monitor backend is Fanotify"));

	monitor->fanotify_fd = fanotify_init (FAN_CLOEXEC |
	                                      FAN_CLASS_NOTIF |
	                                      FAN_REPORT_DFID_NAME,
	                                      O_RDONLY);
	if (monitor->fanotify_fd < 0) {
		g_set_error (error,
		             G_IO_ERROR,
		             g_io_error_from_errno (errno),
		             "Could not initialize Fanotify: %m");
		return FALSE;
	}

	if (!get_fanotify_limit (&limit, error))
		return FALSE;

	/* Take up to 80% of available marks */
	monitor->limit = limit * 8 / 10;
	TRACKER_NOTE (MONITORS, g_message ("Setting a limit of %d  Fanotify marks",
	                                   monitor->limit));

	monitor->source = g_unix_fd_source_new (monitor->fanotify_fd,
	                                     G_IO_IN | G_IO_ERR | G_IO_HUP);
	g_source_set_callback (monitor->source,
	                       (GSourceFunc) fanotify_events_cb,
	                       initable, NULL);
	g_source_attach (monitor->source, NULL);

	return initable_parent_iface->init (initable, cancellable, error);
}

static void
tracker_monitor_fanotify_set_enabled (TrackerMonitor *object,
                                      gboolean        enabled)
{
	TrackerMonitorFanotify *monitor = TRACKER_MONITOR_FANOTIFY (object);
	GList *files = NULL;

	g_return_if_fail (TRACKER_IS_MONITOR (monitor));

	/* Don't replace all monitors if we are already
	 * enabled/disabled.
	 */
	if (monitor->enabled == enabled) {
		return;
	}

	monitor->enabled = enabled;
	g_object_notify (G_OBJECT (monitor), "enabled");

	/* Get the monitored files, and re-add them all */
	files = g_hash_table_get_keys (monitor->monitored_dirs);
	g_list_foreach (files, (GFunc) g_object_ref, NULL);
	g_hash_table_remove_all (monitor->handles);
	g_hash_table_remove_all (monitor->monitored_dirs);

	while (files) {
		GFile *file;

		file = files->data;
		tracker_monitor_add (TRACKER_MONITOR (monitor), file);
		files = g_list_remove (files, file);
		g_object_unref (file);
	}

	TRACKER_MONITOR_CLASS (tracker_monitor_fanotify_parent_class)->set_enabled (object,
                                                                                    enabled);
}

static void
tracker_monitor_fanotify_initable_iface_init (GInitableIface *iface)
{
	initable_parent_iface = g_type_interface_peek_parent (iface);

	iface->init = tracker_monitor_fanotify_initable_init;
}

static void
tracker_monitor_fanotify_finalize (GObject *object)
{
	TrackerMonitorFanotify *monitor = TRACKER_MONITOR_FANOTIFY (object);

	if (monitor->source) {
		g_source_destroy (monitor->source);
		g_source_unref (monitor->source);
	}

	g_hash_table_unref (monitor->monitored_dirs);
	g_hash_table_unref (monitor->handles);
	g_hash_table_unref (monitor->cached_events);
	g_clear_object (&monitor->moved_file);

	G_OBJECT_CLASS (tracker_monitor_fanotify_parent_class)->finalize (object);
}

static void
tracker_monitor_fanotify_set_property (GObject      *object,
                                       guint         prop_id,
                                       const GValue *value,
                                       GParamSpec   *pspec)
{
	switch (prop_id) {
	case PROP_ENABLED:
		tracker_monitor_set_enabled (TRACKER_MONITOR (object),
		                             g_value_get_boolean (value));
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
	}
}

static void
tracker_monitor_fanotify_get_property (GObject      *object,
                                       guint         prop_id,
                                       GValue       *value,
                                       GParamSpec   *pspec)
{
	TrackerMonitorFanotify *monitor = TRACKER_MONITOR_FANOTIFY (object);

	switch (prop_id) {
	case PROP_ENABLED:
		g_value_set_boolean (value, monitor->enabled);
		break;
	case PROP_LIMIT:
		g_value_set_uint (value, monitor->limit);
		break;
	case PROP_COUNT:
		g_value_set_uint (value, tracker_monitor_get_count (TRACKER_MONITOR (object)));
		break;
	case PROP_IGNORED:
		g_value_set_uint (value, monitor->ignored);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
	}
}

static gboolean
add_mark (TrackerMonitorFanotify *monitor,
          GFile                  *file)
{
	gchar *path;

	path = g_file_get_path (file);

	if (fanotify_mark (monitor->fanotify_fd,
	                   (FAN_MARK_ADD | FAN_MARK_ONLYDIR),
	                   FANOTIFY_EVENTS,
	                   AT_FDCWD,
	                   path) < 0) {
		if (errno == EXDEV)
			g_info ("Could not set up cross-device mark for '%s': %m", path);
		else
			g_warning ("Could not add mark for path '%s': %m", path);

		return FALSE;
	}

	g_free (path);
	return TRUE;
}

static void
remove_mark (TrackerMonitorFanotify *monitor,
             GFile                  *file)
{
	gchar *path;

	path = g_file_get_path (file);

	if (fanotify_mark (monitor->fanotify_fd,
	                   FAN_MARK_REMOVE,
	                   FANOTIFY_EVENTS,
	                   AT_FDCWD,
	                   path) < 0) {
		if (errno != ENOENT)
			g_warning ("Could not remove mark for path '%s': %m", path);
	}

	g_free (path);
}

static MonitoredFile *
monitored_file_new (TrackerMonitorFanotify *monitor,
                    GFile                  *file)
{
	MonitoredFile *data;
	gchar *path;
	struct statfs buf;
	int mntid;
	gboolean mark_added = FALSE;

	path = g_file_get_path (file);

	if (statfs (path, &buf) < 0) {
		if (errno != ENOENT)
			g_warning ("Could not get filesystem ID for %s: %m", path);
		g_free (path);
		return NULL;
	}

retry:
	/* We need to try different sizes for the file_handle data */
	data = g_slice_alloc0 (sizeof (MonitoredFile) + monitor->file_handle_payload);
	data->handle.handle.handle_bytes = monitor->file_handle_payload;

	if (name_to_handle_at (AT_FDCWD, path,
	                       (void *) &data->handle.handle,
	                       &mntid,
	                       0) < 0) {
		if (errno == EOVERFLOW) {
			ssize_t payload;

			/* The payload is not big enough to hold a file_handle,
			 * in this case we get the ideal handle data size, so
			 * fetch that and retry.
			 */
			payload = data->handle.handle.handle_bytes;
			g_slice_free1 (sizeof (MonitoredFile) + monitor->file_handle_payload, data);
			monitor->file_handle_payload = payload;
			goto retry;
		} else if (errno != ENOENT) {
			g_warning ("Could not get file handle for '%s': %m", path);
		}

		g_slice_free1 (sizeof (MonitoredFile) +
			       monitor->file_handle_payload, data);
		g_free (path);
		return NULL;
	}

	data->file = g_object_ref (file);
	data->monitor = monitor;
	memcpy (&data->handle.fsid, &buf.f_fsid, sizeof(fsid_t));
	mark_added = add_mark (monitor, file);
	g_free (path);

	if (!mark_added) {
		g_object_unref (data->file);
		g_slice_free1 (sizeof (MonitoredFile) +
			       data->handle.handle.handle_bytes, data);
		return NULL;
	}

	data->handle_bytes = create_bytes_for_handle (&data->handle);

	return data;
}

static void
monitored_file_free (MonitoredFile *data)
{
	if (!data)
		return;

	g_bytes_unref (data->handle_bytes);
	remove_mark (data->monitor, data->file);
	g_object_unref (data->file);
	g_slice_free1 (sizeof (MonitoredFile) +
	               data->handle.handle.handle_bytes, data);
}

static gboolean
tracker_monitor_fanotify_add (TrackerMonitor *object,
                              GFile          *file)
{
	TrackerMonitorFanotify *monitor = TRACKER_MONITOR_FANOTIFY (object);
	MonitoredFile *data;

	if (g_hash_table_contains (monitor->monitored_dirs, file))
		return TRUE;

	if (g_hash_table_size (monitor->monitored_dirs) > monitor->limit) {
		monitor->ignored++;
		return FALSE;
	}

	TRACKER_NOTE (MONITORS, g_message ("Added monitor for path:'%s', total monitors:%d",
	                                   g_file_peek_path (file),
	                                   g_hash_table_size (monitor->monitored_dirs)));

	if (monitor->enabled) {
		data = monitored_file_new (monitor, file);
		if (!data) {
			/* If we cannot create fanotify handles (e.g. EXDEV on
			 * btrfs), fall back to inotify.
			 */
			return TRACKER_MONITOR_CLASS (tracker_monitor_fanotify_parent_class)->add (object,
			                                                                           file);
		}

		g_hash_table_insert (monitor->monitored_dirs, g_object_ref (data->file), data);
		g_hash_table_insert (monitor->handles, data->handle_bytes, data);
	} else {
		g_hash_table_insert (monitor->monitored_dirs, g_object_ref (file), NULL);
	}

	return TRUE;
}

static gboolean
tracker_monitor_fanotify_remove (TrackerMonitor *object,
                                 GFile          *file)
{
	TrackerMonitorFanotify *monitor = TRACKER_MONITOR_FANOTIFY (object);
	MonitoredFile *data;

	data = g_hash_table_lookup (monitor->monitored_dirs, file);
	if (data) {
		g_hash_table_remove (monitor->handles, data->handle_bytes);
		TRACKER_NOTE (MONITORS, g_message ("Removed monitor for path:'%s', total monitors:%d",
		                                   g_file_peek_path (file),
		                                   g_hash_table_size (monitor->monitored_dirs) - 1));
	}

	if (g_hash_table_remove (monitor->monitored_dirs, file))
		return TRUE;

	return TRACKER_MONITOR_CLASS (tracker_monitor_fanotify_parent_class)->remove (object,
	                                                                              file);
}

/* If @is_strict is %TRUE, return %TRUE iff @file is a child of @prefix.
 * If @is_strict is %FALSE, additionally return %TRUE if @file equals @prefix.
 */
static gboolean
file_has_maybe_strict_prefix (GFile    *file,
                              GFile    *prefix,
                              gboolean  is_strict)
{
	return (g_file_has_prefix (file, prefix) ||
	        (!is_strict && g_file_equal (file, prefix)));
}

static gboolean
tracker_monitor_fanotify_remove_recursively (TrackerMonitor *object,
                                             GFile          *file,
                                             gboolean        only_children)
{
	TrackerMonitorFanotify *monitor = TRACKER_MONITOR_FANOTIFY (object);
	MonitoredFile *data;
	GHashTableIter iter;
	guint items_removed = 0;
	GFile *f;
	gchar *uri;

	if (!g_hash_table_contains (monitor->monitored_dirs, file)) {
		return TRACKER_MONITOR_CLASS (tracker_monitor_fanotify_parent_class)->remove_recursively (object,
		                                                                                          file,
		                                                                                          only_children);
	}

	g_hash_table_iter_init (&iter, monitor->monitored_dirs);
	while (g_hash_table_iter_next (&iter, (gpointer *) &f, (gpointer *) &data)) {
		if (!file_has_maybe_strict_prefix (f, file, only_children))
			continue;

		if (data)
			g_hash_table_remove (monitor->handles, data->handle_bytes);
		g_hash_table_iter_remove (&iter);
		items_removed++;
	}

	uri = g_file_get_uri (file);
	TRACKER_NOTE (MONITORS,
	              g_message ("Removed all monitors %srecursively for path:'%s', )"
	                         "total monitors:%d",
	                         only_children ? "(except top level) " : "",
	                         uri, g_hash_table_size (monitor->monitored_dirs)));
	g_free (uri);

	return items_removed > 0;
}

static gboolean
tracker_monitor_fanotify_move (TrackerMonitor *object,
                               GFile          *old_file,
                               GFile          *new_file)
{
	TrackerMonitorFanotify *monitor = TRACKER_MONITOR_FANOTIFY (object);
	MonitoredFile *data;
	GHashTableIter iter;
	gchar *old_prefix;
	gpointer iter_file;
	guint items_moved = 0;
	GList *files = NULL;
	GFile *f;

	if (!g_hash_table_contains (monitor->monitored_dirs, old_file)) {
		return TRACKER_MONITOR_CLASS (tracker_monitor_fanotify_parent_class)->move (object,
		                                                                            old_file,
		                                                                            new_file);
	}

	old_prefix = g_file_get_path (old_file);

	/* Find out which subdirectories should have a file monitor added */
	g_hash_table_iter_init (&iter, monitor->monitored_dirs);
	while (g_hash_table_iter_next (&iter, &iter_file, (gpointer *) &data)) {
		gchar *old_path, *new_path;
		gchar *new_prefix;
		gchar *p;

		if (!file_has_maybe_strict_prefix (iter_file, old_file, FALSE))
			continue;

		old_path = g_file_get_path (iter_file);
		p = strstr (old_path, old_prefix);

		if (!p || strcmp (p, old_prefix) == 0) {
			g_free (old_path);
			continue;
		}

		/* Move to end of prefix */
		p += strlen (old_prefix) + 1;

		/* Check this is not the end of the string */
		if (*p == '\0') {
			g_free (old_path);
			continue;
		}

		new_prefix = g_file_get_path (new_file);
		new_path = g_build_path (G_DIR_SEPARATOR_S, new_prefix, p, NULL);
		g_free (new_prefix);

		f = g_file_new_for_path (new_path);
		g_free (new_path);

		files = g_list_prepend (files, g_object_ref (f));
		if (data)
			g_hash_table_remove (monitor->handles, data->handle_bytes);
		g_hash_table_iter_remove (&iter);

		g_object_unref (f);
		g_free (old_path);
		items_moved++;
	}

	while (files) {
		f = files->data;
		tracker_monitor_fanotify_add (object, f);
		files = g_list_remove (files, files->data);
		g_object_unref (f);
	}

	g_free (old_prefix);

	return items_moved > 0;
}

static gboolean
tracker_monitor_fanotify_is_watched (TrackerMonitor *object,
                                     GFile          *file)
{
	TrackerMonitorFanotify *monitor = TRACKER_MONITOR_FANOTIFY (object);

	if (!monitor->enabled)
		return FALSE;

	if (g_hash_table_contains (monitor->monitored_dirs, file)) {
		return TRUE;
	} else {
		return TRACKER_MONITOR_CLASS (tracker_monitor_fanotify_parent_class)->is_watched (object,
		                                                                                  file);
	}
}

static guint
tracker_monitor_fanotify_get_count (TrackerMonitor *object)
{
	TrackerMonitorFanotify *monitor = TRACKER_MONITOR_FANOTIFY (object);
	guint count;

	count = g_hash_table_size (monitor->monitored_dirs);
	count += TRACKER_MONITOR_CLASS (tracker_monitor_fanotify_parent_class)->get_count (object);

	return count;
}

static void
tracker_monitor_fanotify_class_init (TrackerMonitorFanotifyClass *klass)
{
	TrackerMonitorClass *monitor_class;
	GObjectClass *object_class;

	object_class = G_OBJECT_CLASS (klass);
	monitor_class = TRACKER_MONITOR_CLASS (klass);

	object_class->finalize = tracker_monitor_fanotify_finalize;
	object_class->set_property = tracker_monitor_fanotify_set_property;
	object_class->get_property = tracker_monitor_fanotify_get_property;

	monitor_class->add = tracker_monitor_fanotify_add;
	monitor_class->remove = tracker_monitor_fanotify_remove;
	monitor_class->remove_recursively = tracker_monitor_fanotify_remove_recursively;
	monitor_class->move = tracker_monitor_fanotify_move;
	monitor_class->is_watched = tracker_monitor_fanotify_is_watched;
	monitor_class->set_enabled = tracker_monitor_fanotify_set_enabled;
	monitor_class->get_count = tracker_monitor_fanotify_get_count;

	g_object_class_override_property (object_class, PROP_ENABLED, "enabled");
	g_object_class_override_property (object_class, PROP_LIMIT, "limit");
	g_object_class_override_property (object_class, PROP_COUNT, "count");
	g_object_class_override_property (object_class, PROP_IGNORED, "ignored");
}

static void
tracker_monitor_fanotify_init (TrackerMonitorFanotify *monitor)
{
	/* By default we enable monitoring */
	monitor->enabled = TRUE;

	monitor->monitored_dirs =
		g_hash_table_new_full (g_file_hash,
		                       (GEqualFunc) g_file_equal,
		                       (GDestroyNotify) g_object_unref,
		                       (GDestroyNotify) monitored_file_free);
	monitor->cached_events =
		g_hash_table_new_full (g_file_hash,
		                       (GEqualFunc) g_file_equal,
		                       NULL,
		                       (GDestroyNotify) monitor_event_free);

	monitor->handles = g_hash_table_new (g_bytes_hash, g_bytes_equal);
}
