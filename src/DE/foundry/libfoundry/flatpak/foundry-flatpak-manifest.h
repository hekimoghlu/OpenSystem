/* foundry-flatpak-manifest.h
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

#include "foundry-flatpak-modules.h"
#include "foundry-flatpak-options.h"
#include "foundry-flatpak-serializable.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_FLATPAK_MANIFEST (foundry_flatpak_manifest_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryFlatpakManifest, foundry_flatpak_manifest, FOUNDRY, FLATPAK_MANIFEST, FoundryFlatpakSerializable)

FOUNDRY_AVAILABLE_IN_ALL
FoundryFlatpakModules  *foundry_flatpak_manifest_dup_modules         (FoundryFlatpakManifest *self);
FOUNDRY_AVAILABLE_IN_ALL
char                  **foundry_flatpak_manifest_dup_finish_args     (FoundryFlatpakManifest *self);
FOUNDRY_AVAILABLE_IN_ALL
char                   *foundry_flatpak_manifest_dup_command         (FoundryFlatpakManifest *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryFlatpakOptions  *foundry_flatpak_manifest_dup_build_options   (FoundryFlatpakManifest *self);
FOUNDRY_AVAILABLE_IN_ALL
char                   *foundry_flatpak_manifest_dup_id              (FoundryFlatpakManifest *self);
FOUNDRY_AVAILABLE_IN_ALL
char                   *foundry_flatpak_manifest_dup_sdk             (FoundryFlatpakManifest *self);
FOUNDRY_AVAILABLE_IN_ALL
char                   *foundry_flatpak_manifest_dup_runtime         (FoundryFlatpakManifest *self);
FOUNDRY_AVAILABLE_IN_ALL
char                   *foundry_flatpak_manifest_dup_runtime_version (FoundryFlatpakManifest *self);

G_END_DECLS
