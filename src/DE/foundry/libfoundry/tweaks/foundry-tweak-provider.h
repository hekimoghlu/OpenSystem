/* foundry-tweak-provider.h
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TWEAK_PROVIDER (foundry_tweak_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryTweakProvider, foundry_tweak_provider, FOUNDRY, TWEAK_PROVIDER, FoundryContextual)

struct _FoundryTweakProviderClass
{
  FoundryContextualClass parent_class;

  DexFuture *(*load)   (FoundryTweakProvider *self);
  DexFuture *(*unload) (FoundryTweakProvider *self);

  /*< private >*/
  gpointer _reserved[8];
};

FOUNDRY_AVAILABLE_IN_ALL
guint foundry_tweak_provider_register   (FoundryTweakProvider   *self,
                                         const char             *gettext_package,
                                         const char             *base_path,
                                         const FoundryTweakInfo *info,
                                         guint                   n_infos,
                                         const char * const     *environment);
FOUNDRY_AVAILABLE_IN_ALL
void  foundry_tweak_provider_unregister (FoundryTweakProvider   *self,
                                         guint                   registration);

G_END_DECLS
