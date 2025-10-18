/* foundry-tweak-tree.h
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

#include <libdex.h>

#include "foundry-types.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TWEAK_TREE (foundry_tweak_tree_get_type())

G_DECLARE_FINAL_TYPE (FoundryTweakTree, foundry_tweak_tree, FOUNDRY, TWEAK_TREE, GObject)

FoundryTweakTree *foundry_tweak_tree_new        (FoundryContext         *context);
void              foundry_tweak_tree_add        (FoundryTweakTree       *self,
                                                 FoundryTweak           *tweak);
void              foundry_tweak_tree_remove     (FoundryTweakTree       *self,
                                                 FoundryTweak           *tweak);
guint             foundry_tweak_tree_register   (FoundryTweakTree       *self,
                                                 const char             *gettext_domain,
                                                 const char             *base_path,
                                                 const FoundryTweakInfo *info,
                                                 guint                   n_info,
                                                 const char * const     *environment);
void              foundry_tweak_tree_unregister (FoundryTweakTree       *self,
                                                 guint                   registration);
DexFuture        *foundry_tweak_tree_list       (FoundryTweakTree       *self,
                                                 const char             *path);

G_END_DECLS
