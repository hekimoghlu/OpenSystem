/* foundry-flatpak-manifest-loader.h
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

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_FLATPAK_MANIFEST_LOADER (foundry_flatpak_manifest_loader_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryFlatpakManifestLoader, foundry_flatpak_manifest_loader, FOUNDRY, FLATPAK_MANIFEST_LOADER, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryFlatpakManifestLoader *foundry_flatpak_manifest_loader_new          (GFile                        *file);
FOUNDRY_AVAILABLE_IN_ALL
GFile                        *foundry_flatpak_manifest_loader_dup_file     (FoundryFlatpakManifestLoader *self);
FOUNDRY_AVAILABLE_IN_ALL
GFile                        *foundry_flatpak_manifest_loader_dup_base_dir (FoundryFlatpakManifestLoader *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                    *foundry_flatpak_manifest_loader_load         (FoundryFlatpakManifestLoader *self);

G_END_DECLS
