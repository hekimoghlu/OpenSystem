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

#ifndef __LIBTRACKER_MINER_OBJECT_H__
#define __LIBTRACKER_MINER_OBJECT_H__

#include <glib-object.h>
#include <gio/gio.h>

#include <tinysparql.h>

G_BEGIN_DECLS

/* Common definitions for all miners */
/**
 * TRACKER_MINER_DBUS_INTERFACE:
 *
 * The name of the D-Bus interface to use for all data miners that
 * inter-operate with Tracker.
 *
 * Since: 0.8
 **/
#define TRACKER_MINER_DBUS_INTERFACE   "org.freedesktop.Tracker3.Miner"

/**
 * TRACKER_MINER_DBUS_NAME_PREFIX:
 *
 * D-Bus name prefix to use for all data miners. This allows custom
 * miners to be written using @TRACKER_MINER_DBUS_NAME_PREFIX + "Files" for
 * example and would show up on D-Bus under
 * &quot;org.freedesktop.Tracker3.Miner.Files&quot;.
 *
 * Since: 0.8
 **/
#define TRACKER_MINER_DBUS_NAME_PREFIX "org.freedesktop.Tracker3.Miner."

/**
 * TRACKER_MINER_DBUS_PATH_PREFIX:
 *
 * D-Bus path prefix to use for all data miners. This allows custom
 * miners to be written using @TRACKER_MINER_DBUS_PATH_PREFIX + "Files" for
 * example and would show up on D-Bus under
 * &quot;/org/freedesktop/Tracker3/Miner/Files&quot;.
 *
 * Since: 0.8
 **/
#define TRACKER_MINER_DBUS_PATH_PREFIX "/org/freedesktop/Tracker3/Miner/"

#define TRACKER_TYPE_MINER tracker_miner_get_type()
G_DECLARE_DERIVABLE_TYPE (TrackerMiner, tracker_miner, TRACKER, MINER, GObject)

/**
 * TRACKER_MINER_ERROR:
 *
 * Returns the @GQuark used for #GErrors and for @TrackerMiner
 * implementations. This calls tracker_miner_error_quark().
 *
 * Since: 0.8
 **/
#define TRACKER_MINER_ERROR        tracker_miner_error_quark()

struct _TrackerMinerClass {
	GObjectClass parent_class;

	/* signals */
	void (* started)            (TrackerMiner *miner);
	void (* stopped)            (TrackerMiner *miner);

	void (* paused)             (TrackerMiner *miner);
	void (* resumed)            (TrackerMiner *miner);

	void (* progress)           (TrackerMiner *miner,
	                             const gchar  *status,
	                             gdouble       progress,
	                             gint          remaining_time);
};

typedef enum {
	TRACKER_MINER_ERROR_PAUSED_ALREADY,
	TRACKER_MINER_ERROR_INVALID_COOKIE
} TrackerMinerError;

GQuark                   tracker_miner_error_quark         (void);

void                     tracker_miner_start               (TrackerMiner         *miner);
void                     tracker_miner_stop                (TrackerMiner         *miner);
gboolean                 tracker_miner_is_started          (TrackerMiner         *miner);
gboolean                 tracker_miner_is_paused           (TrackerMiner         *miner);

void                     tracker_miner_pause               (TrackerMiner         *miner);
gboolean                 tracker_miner_resume              (TrackerMiner         *miner);

TrackerSparqlConnection *tracker_miner_get_connection      (TrackerMiner         *miner);

G_END_DECLS

#endif /* __LIBTRACKER_MINER_OBJECT_H__ */
