/* foundry-device-chassis.h
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include <glib-object.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEVICE_CHASSIS (foundry_device_chassis_get_type())

typedef enum _FoundryDeviceChassis
{
  FOUNDRY_DEVICE_CHASSIS_WORKSTATION,
  FOUNDRY_DEVICE_CHASSIS_HANDSET,
  FOUNDRY_DEVICE_CHASSIS_TABLET,
  FOUNDRY_DEVICE_CHASSIS_OTHER,

  /* Not part of ABI */
  FOUNDRY_DEVICE_CHASSIS_LAST,
} FoundryDeviceChassis;

FOUNDRY_AVAILABLE_IN_ALL
GType foundry_device_chassis_get_type (void) G_GNUC_CONST;

G_END_DECLS
