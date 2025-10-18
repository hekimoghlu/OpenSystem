/*
 * Copyright (C) 2021, Codethink Ltd
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 *
 * Author: Sam Thursfield <sam@afuera.me.uk>
 */

#ifndef __MOCK_MOUNT_H__
#define __MOCK_MOUNT_H__

#include <glib-object.h>
#include <gio/gio.h>
#include <gio/gunixmounts.h>

#include "mock-volume-monitor.h"

G_BEGIN_DECLS

#define MOCK_TYPE_MOUNT  (mock_mount_get_type ())
G_DECLARE_FINAL_TYPE (MockMount, mock_mount, MOCK, MOUNT, GObject)


MockMount        *mock_mount_new           (MockVolumeMonitor *monitor,
                                            MockVolume        *volume,
                                            const gchar       *name,
                                            GFile             *root);
void              mock_mount_unmounted     (MockMount         *mount);

void              mock_mount_set_volume    (MockMount         *mount,
                                            MockVolume        *volume);
void              mock_mount_unset_volume  (MockMount         *mount,
                                            MockVolume        *volume);
MockVolume       *mock_mount_get_volume    (MockMount         *mount);

G_END_DECLS

#endif /* __MOCK_MOUNT_H__ */
