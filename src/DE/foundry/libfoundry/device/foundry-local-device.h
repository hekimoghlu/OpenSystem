/* foundry-local-device.h
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "foundry-device.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LOCAL_DEVICE (foundry_local_device_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryLocalDevice, foundry_local_device, FOUNDRY, LOCAL_DEVICE, FoundryDevice)

FOUNDRY_AVAILABLE_IN_ALL
FoundryDevice *foundry_local_device_new      (FoundryContext *context);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDevice *foundry_local_device_new_full (FoundryContext *context,
                                              const char     *id,
                                              const char     *title,
                                              FoundryTriplet *triplet);

G_END_DECLS
