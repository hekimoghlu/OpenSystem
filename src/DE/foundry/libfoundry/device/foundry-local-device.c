/*
 * foundry-local-device.c
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

#include "config.h"

#include <glib/gi18n-lib.h>

#include "foundry-local-device.h"
#include "foundry-local-device-info-private.h"
#include "foundry-triplet.h"

struct _FoundryLocalDevice
{
  FoundryDevice parent_instance;
  char *id;
  char *title;
  FoundryTriplet *triplet;
};

G_DEFINE_FINAL_TYPE (FoundryLocalDevice, foundry_local_device, FOUNDRY_TYPE_DEVICE)

static char *
foundry_local_device_dup_id (FoundryDevice *self)
{
  return g_strdup (FOUNDRY_LOCAL_DEVICE (self)->id);
}

static DexFuture *
foundry_local_device_load_info (FoundryDevice *device)
{
  FoundryLocalDevice *self = (FoundryLocalDevice *)device;
  g_autoptr(FoundryDeviceInfo) device_info = NULL;

  g_assert (FOUNDRY_IS_LOCAL_DEVICE (self));

  device_info = foundry_local_device_info_new (device,
                                               self->title,
                                               FOUNDRY_DEVICE_CHASSIS_WORKSTATION,
                                               self->triplet);

  return dex_future_new_take_object (g_steal_pointer (&device_info));
}

static void
foundry_local_device_finalize (GObject *object)
{
  FoundryLocalDevice *self = (FoundryLocalDevice *)object;

  g_clear_pointer (&self->id, g_free);
  g_clear_pointer (&self->title, g_free);
  g_clear_pointer (&self->triplet, foundry_triplet_unref);

  G_OBJECT_CLASS (foundry_local_device_parent_class)->finalize (object);
}

static void
foundry_local_device_class_init (FoundryLocalDeviceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDeviceClass *device_class = FOUNDRY_DEVICE_CLASS (klass);

  object_class->finalize = foundry_local_device_finalize;

  device_class->dup_id = foundry_local_device_dup_id;
  device_class->load_info = foundry_local_device_load_info;
}

static void
foundry_local_device_init (FoundryLocalDevice *self)
{
}

FoundryDevice *
foundry_local_device_new (FoundryContext *context)
{
  g_autoptr(FoundryTriplet) triplet = NULL;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);

  triplet = foundry_triplet_new_from_system ();

  return foundry_local_device_new_full (context, "native", _("My Computer"), triplet);
}

FoundryDevice *
foundry_local_device_new_full (FoundryContext *context,
                               const char     *id,
                               const char     *title,
                               FoundryTriplet *triplet)
{
  FoundryLocalDevice *self;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (triplet != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_LOCAL_DEVICE,
                       "context", context,
                       NULL);

  self->id = g_strdup (id);
  self->title = g_strdup (title);
  self->triplet = foundry_triplet_ref (triplet);

  return FOUNDRY_DEVICE (self);
}
