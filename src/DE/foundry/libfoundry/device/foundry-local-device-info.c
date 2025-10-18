/* foundry-local-device-info.c
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

#include "config.h"

#include "foundry-local-device-info-private.h"

struct _FoundryLocalDeviceInfo
{
  FoundryDeviceInfo parent_instance;
  char *id;
  char *name;
  FoundryTriplet *triplet;
  FoundryDeviceChassis chassis;
};

G_DEFINE_FINAL_TYPE (FoundryLocalDeviceInfo, foundry_local_device_info, FOUNDRY_TYPE_DEVICE_INFO)

static char *
foundry_local_device_info_dup_id (FoundryDeviceInfo *device_info)
{
  return g_strdup (FOUNDRY_LOCAL_DEVICE_INFO (device_info)->id);
}

static char *
foundry_local_device_info_dup_name (FoundryDeviceInfo *device_info)
{
  return g_strdup (FOUNDRY_LOCAL_DEVICE_INFO (device_info)->name);
}

static FoundryTriplet *
foundry_local_device_info_dup_triplet (FoundryDeviceInfo *device_info)
{
  return foundry_triplet_ref (FOUNDRY_LOCAL_DEVICE_INFO (device_info)->triplet);
}

static FoundryDeviceChassis
foundry_local_device_info_get_chassis (FoundryDeviceInfo *device_info)
{
  return FOUNDRY_LOCAL_DEVICE_INFO (device_info)->chassis;
}

static void
foundry_local_device_info_finalize (GObject *object)
{
  FoundryLocalDeviceInfo *self = (FoundryLocalDeviceInfo *)object;

  g_clear_pointer (&self->id, g_free);
  g_clear_pointer (&self->name, g_free);
  g_clear_pointer (&self->triplet, foundry_triplet_unref);

  G_OBJECT_CLASS (foundry_local_device_info_parent_class)->finalize (object);
}

static void
foundry_local_device_info_class_init (FoundryLocalDeviceInfoClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDeviceInfoClass *device_info_class = FOUNDRY_DEVICE_INFO_CLASS (klass);

  object_class->finalize = foundry_local_device_info_finalize;

  device_info_class->dup_id = foundry_local_device_info_dup_id;
  device_info_class->dup_name = foundry_local_device_info_dup_name;
  device_info_class->dup_triplet = foundry_local_device_info_dup_triplet;
  device_info_class->get_chassis = foundry_local_device_info_get_chassis;
}

static void
foundry_local_device_info_init (FoundryLocalDeviceInfo *self)
{
}

FoundryDeviceInfo *
foundry_local_device_info_new (FoundryDevice        *device,
                               const char           *name,
                               FoundryDeviceChassis  chassis,
                               FoundryTriplet       *triplet)
{
  FoundryLocalDeviceInfo *self;

  g_return_val_if_fail (FOUNDRY_IS_DEVICE (device), NULL);
  g_return_val_if_fail (name != NULL, NULL);
  g_return_val_if_fail (triplet != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_LOCAL_DEVICE_INFO,
                       "device", device,
                       NULL);
  self->id = foundry_device_dup_id (device);
  self->name = g_strdup (name);
  self->chassis = chassis;
  self->triplet = foundry_triplet_ref (triplet);

  return FOUNDRY_DEVICE_INFO (self);
}
