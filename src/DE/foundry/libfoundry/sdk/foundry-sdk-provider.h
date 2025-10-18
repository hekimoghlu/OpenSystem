/* foundry-sdk-provider.h
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

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SDK_PROVIDER (foundry_sdk_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundrySdkProvider, foundry_sdk_provider, FOUNDRY, SDK_PROVIDER, FoundryContextual)

struct _FoundrySdkProviderClass
{
  FoundryContextualClass parent_class;

  char      *(*dup_name)   (FoundrySdkProvider *self);
  DexFuture *(*load)       (FoundrySdkProvider *self);
  DexFuture *(*unload)     (FoundrySdkProvider *self);
  DexFuture *(*find_by_id) (FoundrySdkProvider *self,
                            const char         *sdk_id);

  /*< private >*/
  gpointer _reserved[8];
};

FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_sdk_provider_dup_name    (FoundrySdkProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_sdk_provider_sdk_added   (FoundrySdkProvider *self,
                                             FoundrySdk         *sdk);
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_sdk_provider_sdk_removed (FoundrySdkProvider *self,
                                             FoundrySdk         *sdk);
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_sdk_provider_merge       (FoundrySdkProvider *self,
                                             GPtrArray          *sdks);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_sdk_provider_find_by_id  (FoundrySdkProvider *self,
                                             const char         *sdk_id) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
