/* foundry-device-info.h
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

#include <glib-object.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEVICE_INFO (foundry_device_info_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDeviceInfo, foundry_device_info, FOUNDRY, DEVICE_INFO, GObject)

struct _FoundryDeviceInfoClass
{
  GObjectClass           parent_class;

  char                 *(*dup_id)      (FoundryDeviceInfo *self);
  char                 *(*dup_name)    (FoundryDeviceInfo *self);
  FoundryTriplet       *(*dup_triplet) (FoundryDeviceInfo *self);
  FoundryDeviceChassis  (*get_chassis) (FoundryDeviceInfo *self);

  /*< private >*/
  gpointer _reserved[11];
};

FOUNDRY_AVAILABLE_IN_ALL
gboolean              foundry_device_info_get_active  (FoundryDeviceInfo *self);
FOUNDRY_AVAILABLE_IN_ALL
char                 *foundry_device_info_dup_id      (FoundryDeviceInfo *self);
FOUNDRY_AVAILABLE_IN_ALL
char                 *foundry_device_info_dup_name    (FoundryDeviceInfo *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTriplet       *foundry_device_info_dup_triplet (FoundryDeviceInfo *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDeviceChassis  foundry_device_info_get_chassis (FoundryDeviceInfo *self);

G_END_DECLS
