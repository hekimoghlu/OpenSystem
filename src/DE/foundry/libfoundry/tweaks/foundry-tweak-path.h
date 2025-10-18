/* foundry-tweak-path.h
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

G_BEGIN_DECLS

FoundryTweakPath *foundry_tweak_path_new           (const char             *path);
void              foundry_tweak_path_free          (FoundryTweakPath       *self);
char             *foundry_tweak_path_dup_path      (const FoundryTweakPath *self);
gboolean          foundry_tweak_path_has_prefix    (const FoundryTweakPath *self,
                                                    const FoundryTweakPath *other);
gboolean          foundry_tweak_path_equal         (const FoundryTweakPath *self,
                                                    const FoundryTweakPath *other);
guint             foundry_tweak_path_get_depth     (const FoundryTweakPath *self,
                                                    const FoundryTweakPath *other);
int               foundry_tweak_path_compute_depth (const FoundryTweakPath *self,
                                                    const FoundryTweakPath *other);
int               foundry_tweak_path_compare       (const FoundryTweakPath *self,
                                                    const FoundryTweakPath *other);
FoundryTweakPath *foundry_tweak_path_push          (const FoundryTweakPath *self,
                                                    const char             *subpath);
FoundryTweakPath *foundry_tweak_path_pop           (const FoundryTweakPath *self);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (FoundryTweakPath, foundry_tweak_path_free)

G_END_DECLS
