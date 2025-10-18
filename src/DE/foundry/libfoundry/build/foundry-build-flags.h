/* foundry-build-flags.h
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

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_BUILD_FLAGS (foundry_build_flags_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryBuildFlags, foundry_build_flags, FOUNDRY, BUILD_FLAGS, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryBuildFlags  *foundry_build_flags_new           (const char * const *flags,
                                                       const char         *directory);
FOUNDRY_AVAILABLE_IN_ALL
char              **foundry_build_flags_dup_flags     (FoundryBuildFlags  *self);
FOUNDRY_AVAILABLE_IN_ALL
char               *foundry_build_flags_dup_directory (FoundryBuildFlags  *self);

G_END_DECLS
