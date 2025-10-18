/* foundry-local-device-info.h
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "foundry-device.h"
#include "foundry-device-chassis.h"
#include "foundry-device-info.h"
#include "foundry-triplet.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LOCAL_DEVICE_INFO (foundry_local_device_info_get_type())

G_DECLARE_FINAL_TYPE (FoundryLocalDeviceInfo, foundry_local_device_info, FOUNDRY, LOCAL_DEVICE_INFO, FoundryDeviceInfo)

FoundryDeviceInfo *foundry_local_device_info_new (FoundryDevice        *device,
                                                  const char           *name,
                                                  FoundryDeviceChassis  chassis,
                                                  FoundryTriplet       *triplet);

G_END_DECLS
