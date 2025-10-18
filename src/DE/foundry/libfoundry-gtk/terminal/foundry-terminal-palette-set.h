/* foundry-terminal-palette-set.h
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

#include "foundry-terminal-palette.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TERMINAL_PALETTE_SET (foundry_terminal_palette_set_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryTerminalPaletteSet, foundry_terminal_palette_set, FOUNDRY, TERMINAL_PALETTE_SET, GObject)

FOUNDRY_AVAILABLE_IN_ALL
DexFuture              *foundry_terminal_palette_set_new       (GBytes                    *bytes);
FOUNDRY_AVAILABLE_IN_ALL
char                   *foundry_terminal_palette_set_dup_title (FoundryTerminalPaletteSet *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTerminalPalette *foundry_terminal_palette_set_dup_dark  (FoundryTerminalPaletteSet *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTerminalPalette *foundry_terminal_palette_set_dup_light (FoundryTerminalPaletteSet *self);

G_END_DECLS
