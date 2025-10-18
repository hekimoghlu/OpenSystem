/* foundry-build-addin.h
 *
 * Copyright 2024-2025 Christian Hergert <chergert@redhat.com>
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

#include <libpeas.h>

#include "foundry-contextual.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_BUILD_ADDIN (foundry_build_addin_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryBuildAddin, foundry_build_addin, FOUNDRY, BUILD_ADDIN, FoundryContextual)

struct _FoundryBuildAddinClass
{
  FoundryContextualClass parent_class;

  DexFuture *(*load)   (FoundryBuildAddin *self);
  DexFuture *(*unload) (FoundryBuildAddin *self);

  /*< private >*/
  gpointer _reserved[14];
};

FOUNDRY_AVAILABLE_IN_ALL
FoundryBuildPipeline *foundry_build_addin_dup_pipeline    (FoundryBuildAddin *self);
FOUNDRY_AVAILABLE_IN_ALL
PeasPluginInfo       *foundry_build_addin_dup_plugin_info (FoundryBuildAddin *self);

G_END_DECLS
