/* foundry-flatpak-arch-options.h
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

#include "foundry-flatpak-options.h"
#include "foundry-flatpak-serializable.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_FLATPAK_ARCH_OPTIONS (foundry_flatpak_arch_options_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryFlatpakArchOptions, foundry_flatpak_arch_options, FOUNDRY, FLATPAK_ARCH_OPTIONS, FoundryFlatpakSerializable)

FOUNDRY_AVAILABLE_IN_ALL
char                  **foundry_flatpak_arch_options_dup_arches (FoundryFlatpakArchOptions *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryFlatpakOptions  *foundry_flatpak_arch_options_dup_arch   (FoundryFlatpakArchOptions *self,
                                                                 const char                *arch);

G_END_DECLS
