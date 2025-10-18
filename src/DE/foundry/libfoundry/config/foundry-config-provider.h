/* foundry-config-provider.h
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

#define FOUNDRY_TYPE_CONFIG_PROVIDER (foundry_config_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryConfigProvider, foundry_config_provider, FOUNDRY, CONFIG_PROVIDER, FoundryContextual)

struct _FoundryConfigProviderClass
{
  FoundryContextualClass parent_class;

  char      *(*dup_name) (FoundryConfigProvider *self);
  DexFuture *(*load)     (FoundryConfigProvider *self);
  DexFuture *(*unload)   (FoundryConfigProvider *self);
  DexFuture *(*save)     (FoundryConfigProvider *self);
  DexFuture *(*delete)   (FoundryConfigProvider *self,
                          FoundryConfig         *config);
  DexFuture *(*copy)     (FoundryConfigProvider *self,
                          FoundryConfig         *config);

  /*< private >*/
  gpointer _reserved[10];
};

FOUNDRY_AVAILABLE_IN_ALL
char *foundry_config_provider_dup_name       (FoundryConfigProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
void  foundry_config_provider_config_added   (FoundryConfigProvider *self,
                                              FoundryConfig         *config);
FOUNDRY_AVAILABLE_IN_ALL
void  foundry_config_provider_config_removed (FoundryConfigProvider *self,
                                              FoundryConfig         *config);

G_END_DECLS
