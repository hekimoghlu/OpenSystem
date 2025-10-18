/* foundry-build-target.h
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

#include <glib-object.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_BUILD_TARGET (foundry_build_target_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryBuildTarget, foundry_build_target, FOUNDRY, BUILD_TARGET, GObject)

struct _FoundryBuildTargetClass
{
  GObjectClass parent_class;

  char *(*dup_id)    (FoundryBuildTarget *self);
  char *(*dup_title) (FoundryBuildTarget *self);

  /*< private >*/
  gpointer _reserved[13];
};

FOUNDRY_AVAILABLE_IN_ALL
char *foundry_build_target_dup_id    (FoundryBuildTarget *self);
FOUNDRY_AVAILABLE_IN_ALL
char *foundry_build_target_dup_title (FoundryBuildTarget *self);

G_END_DECLS
