/* foundry-flatpak-modules.h
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

#include "foundry-flatpak-list.h"
#include "foundry-flatpak-module.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_FLATPAK_MODULES (foundry_flatpak_modules_get_type())

FOUNDRY_AVAILABLE_IN_ALL
FOUNDRY_DECLARE_INTERNAL_TYPE (FoundryFlatpakModules, foundry_flatpak_modules, FOUNDRY, FLATPAK_MODULES, FoundryFlatpakList)

FOUNDRY_AVAILABLE_IN_ALL
FoundryFlatpakModule *foundry_flatpak_modules_find_primary (FoundryFlatpakModules *self,
                                                            GFile                 *project_dir);

G_END_DECLS
