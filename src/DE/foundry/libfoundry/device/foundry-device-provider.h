/* foundry-device-provider.h
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

#include "foundry-contextual.h"
#include "foundry-types.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEVICE_PROVIDER (foundry_device_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDeviceProvider, foundry_device_provider, FOUNDRY, DEVICE_PROVIDER, FoundryContextual)

struct _FoundryDeviceProviderClass
{
  FoundryContextualClass parent_class;

  char      *(*dup_name) (FoundryDeviceProvider *self);
  DexFuture *(*load)     (FoundryDeviceProvider *self);
  DexFuture *(*unload)   (FoundryDeviceProvider *self);

  /*< private >*/
  gpointer _reserved[8];
};

FOUNDRY_AVAILABLE_IN_ALL
char *foundry_device_provider_dup_name       (FoundryDeviceProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
void  foundry_device_provider_device_added   (FoundryDeviceProvider *self,
                                              FoundryDevice         *device);
FOUNDRY_AVAILABLE_IN_ALL
void  foundry_device_provider_device_removed (FoundryDeviceProvider *self,
                                              FoundryDevice         *device);

G_END_DECLS

