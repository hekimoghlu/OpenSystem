/*
 * Copyright (C) 2023 Red Hat Inc.

 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 *
 * Author: Carlos Garnacho <carlosg@gnome.org>
 */

#ifndef __TRACKER_FILES_INTERFACE_H__
#define __TRACKER_FILES_INTERFACE_H__

#include <gio/gio.h>

#define TRACKER_TYPE_FILES_INTERFACE (tracker_files_interface_get_type ())
G_DECLARE_FINAL_TYPE (TrackerFilesInterface,
                      tracker_files_interface,
                      TRACKER, FILES_INTERFACE,
                      GObject)

TrackerFilesInterface * tracker_files_interface_new (GDBusConnection *connection);

TrackerFilesInterface * tracker_files_interface_new_with_fd (GDBusConnection *connection,
                                                             int              fd);

int tracker_files_interface_dup_fd (TrackerFilesInterface *files_interface);

void tracker_files_interface_set_priority_graphs (TrackerFilesInterface *files_interface,
                                                  GVariant              *graphs);

#endif /* __TRACKER_FILES_INTERFACE_H__ */
