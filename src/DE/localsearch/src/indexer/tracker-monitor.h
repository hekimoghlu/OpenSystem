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

#ifndef __LIBTRACKER_MINER_MONITOR_H__
#define __LIBTRACKER_MINER_MONITOR_H__

#include <glib-object.h>
#include <gio/gio.h>

G_BEGIN_DECLS

#define TRACKER_TYPE_MONITOR (tracker_monitor_get_type ())
G_DECLARE_DERIVABLE_TYPE (TrackerMonitor, tracker_monitor,
                          TRACKER, MONITOR,
                          GObject)

struct _TrackerMonitorClass {
	GObjectClass parent;

	gboolean (* add) (TrackerMonitor *monitor,
	                  GFile          *file);
	gboolean (* remove) (TrackerMonitor *monitor,
	                     GFile          *file);
	gboolean (* remove_recursively) (TrackerMonitor *monitor,
	                                 GFile          *file,
	                                 gboolean        only_children);
	gboolean (* move) (TrackerMonitor *monitor,
	                   GFile          *src,
	                   GFile          *dst);
	gboolean (* is_watched) (TrackerMonitor *monitor,
	                         GFile          *file);
	void (* set_enabled) (TrackerMonitor *monitor,
	                      gboolean        enabled);
	guint (* get_count) (TrackerMonitor *monitor);
};

GType           tracker_monitor_get_type             (void);

gboolean        tracker_monitor_get_enabled          (TrackerMonitor *monitor);
void            tracker_monitor_set_enabled          (TrackerMonitor *monitor,
                                                      gboolean        enabled);
gboolean        tracker_monitor_add                  (TrackerMonitor *monitor,
                                                      GFile          *file);
gboolean        tracker_monitor_remove               (TrackerMonitor *monitor,
                                                      GFile          *file);
gboolean        tracker_monitor_remove_recursively   (TrackerMonitor *monitor,
                                                      GFile          *file);
gboolean        tracker_monitor_remove_children_recursively (TrackerMonitor *monitor,
                                                             GFile          *file);
gboolean        tracker_monitor_move                 (TrackerMonitor *monitor,
                                                      GFile          *old_file,
                                                      GFile          *new_file);
gboolean        tracker_monitor_is_watched           (TrackerMonitor *monitor,
                                                      GFile          *file);
guint           tracker_monitor_get_count            (TrackerMonitor *monitor);
guint           tracker_monitor_get_ignored          (TrackerMonitor *monitor);
guint           tracker_monitor_get_limit            (TrackerMonitor *monitor);

TrackerMonitor * tracker_monitor_new (GError **error);

G_END_DECLS

#endif /* __LIBTRACKER_MINER_MONITOR_H__ */
