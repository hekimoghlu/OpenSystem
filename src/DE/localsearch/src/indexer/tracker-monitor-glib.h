/*
 * Copyright (C) 2021, Red Hat Inc.
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

#ifndef __LIBTRACKER_MINER_MONITOR_GLIB_H__
#define __LIBTRACKER_MINER_MONITOR_GLIB_H__

#include <glib-object.h>
#include <gio/gio.h>

#include "tracker-monitor.h"

G_BEGIN_DECLS

#define TRACKER_TYPE_MONITOR_GLIB (tracker_monitor_glib_get_type ())
G_DECLARE_DERIVABLE_TYPE (TrackerMonitorGlib, tracker_monitor_glib,
                          TRACKER, MONITOR_GLIB,
                          TrackerMonitor)

struct _TrackerMonitorGlibClass {
	TrackerMonitorClass parent_class;
};

G_END_DECLS

#endif /* __LIBTRACKER_MINER_MONITOR_GLIB_H__ */
