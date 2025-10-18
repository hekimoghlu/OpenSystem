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

#ifndef __MOCK_VOLUME_H__
#define __MOCK_VOLUME_H__

#include <glib-object.h>
#include <gio/gio.h>
#include <gio/gunixmounts.h>

#include "mock-volume-monitor.h"

G_BEGIN_DECLS

#define MOCK_TYPE_VOLUME  (mock_volume_get_type ())
G_DECLARE_FINAL_TYPE (MockVolume, mock_volume, MOCK, VOLUME, GObject)

MockVolume *mock_volume_new         (MockVolumeMonitor   *monitor,
                                     MockDrive           *drive,
                                     const gchar         *name);
void               mock_volume_removed     (MockVolume          *volume);

GUnixMountPoint   *mock_volume_get_mount_point (MockVolume      *volume);

void               mock_volume_set_mount   (MockVolume          *volume,
                                            MockMount           *mount);
void               mock_volume_unset_mount (MockVolume          *volume,
                                            MockMount           *mount);

void               mock_volume_set_drive   (MockVolume          *volume,
                                            MockDrive           *drive);
void               mock_volume_unset_drive (MockVolume          *volume,
                                            MockDrive           *drive);

G_END_DECLS

#endif /* __MOCK_VOLUME_H__ */
