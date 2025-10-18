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

#ifndef __MOCK_VOLUME_MONITOR_H__
#define __MOCK_VOLUME_MONITOR_H__

#include <gio/gio.h>

#define MOCK_TYPE_VOLUME_MONITOR mock_volume_monitor_get_type()
G_DECLARE_FINAL_TYPE (MockVolumeMonitor, mock_volume_monitor, MOCK, VOLUME_MONITOR, GNativeVolumeMonitor)

/* Forward definitions */
typedef struct _MockDrive MockDrive;
typedef struct _MockVolume MockVolume;
typedef struct _MockMount MockMount;

#endif
