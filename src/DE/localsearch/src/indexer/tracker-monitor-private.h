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

#include "tracker-monitor.h"

void tracker_monitor_emit_created (TrackerMonitor *monitor,
                                   GFile          *file,
                                   gboolean        is_directory);
void tracker_monitor_emit_updated (TrackerMonitor *monitor,
                                   GFile          *file,
                                   gboolean        is_directory);
void tracker_monitor_emit_attributes_updated (TrackerMonitor *monitor,
                                              GFile          *file,
                                              gboolean        is_directory);
void tracker_monitor_emit_deleted (TrackerMonitor *monitor,
                                   GFile          *file,
                                   gboolean        is_directory);
void tracker_monitor_emit_moved (TrackerMonitor *monitor,
                                 GFile          *file,
                                 GFile          *other_file,
                                 gboolean        is_directory);
