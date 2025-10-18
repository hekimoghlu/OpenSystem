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


#ifndef __MOCK_DRIVE_H__
#define __MOCK_DRIVE_H__

#include <glib-object.h>
#include <gio/gio.h>
#include <gio/gunixmounts.h>

#include "mock-volume-monitor.h"
#include "mock-volume.h"

G_BEGIN_DECLS

#define MOCK_TYPE_DRIVE  (mock_drive_get_type ())
G_DECLARE_FINAL_TYPE (MockDrive, mock_drive, MOCK, DRIVE, GObject)

MockDrive *mock_drive_new             (MockVolumeMonitor *monitor,
                                       const gchar       *name,
                                       gboolean           removable);
void       mock_drive_disconnected    (MockDrive         *drive);
void       mock_drive_set_volume      (MockDrive         *drive,
                                       MockVolume        *volume);
void       mock_drive_unset_volume    (MockDrive         *drive,
                                       MockVolume        *volume);

G_END_DECLS

#endif /* __MOCK_DRIVE_H__ */
