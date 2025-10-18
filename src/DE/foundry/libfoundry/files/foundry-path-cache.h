/* foundry-path-cache.h
 *
 * Copyright 2022-2024 Christian Hergert <chergert@redhat.com>
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
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_PATH_CACHE (foundry_path_cache_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryPathCache, foundry_path_cache, FOUNDRY, PATH_CACHE, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryPathCache *foundry_path_cache_new      (void);
FOUNDRY_AVAILABLE_IN_ALL
gboolean          foundry_path_cache_lookup   (FoundryPathCache  *self,
                                               const char        *program_name,
                                               char             **program_path);
FOUNDRY_AVAILABLE_IN_ALL
gboolean          foundry_path_cache_contains (FoundryPathCache  *self,
                                               const char        *program_name,
                                               gboolean          *had_program_path);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_path_cache_insert   (FoundryPathCache  *self,
                                               const char        *program_name,
                                               const char        *program_path);

G_END_DECLS
