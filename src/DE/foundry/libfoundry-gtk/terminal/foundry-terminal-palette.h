/* foundry-terminal-palette.h
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

#include <gdk/gdk.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TERMINAL_PALETTE (foundry_terminal_palette_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryTerminalPalette, foundry_terminal_palette, FOUNDRY, TERMINAL_PALETTE, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryTerminalPalette *foundry_terminal_palette_new                   (void);
FOUNDRY_AVAILABLE_IN_ALL
const GdkRGBA          *foundry_terminal_palette_get_colors            (FoundryTerminalPalette *self,
                                                                        guint                  *palette_length);
FOUNDRY_AVAILABLE_IN_ALL
const GdkRGBA          *foundry_terminal_palette_get_foreground        (FoundryTerminalPalette *self);
FOUNDRY_AVAILABLE_IN_ALL
const GdkRGBA          *foundry_terminal_palette_get_background        (FoundryTerminalPalette *self);
FOUNDRY_AVAILABLE_IN_ALL
const GdkRGBA          *foundry_terminal_palette_get_cursor_foreground (FoundryTerminalPalette *self);
FOUNDRY_AVAILABLE_IN_ALL
const GdkRGBA          *foundry_terminal_palette_get_cursor_background (FoundryTerminalPalette *self);

G_END_DECLS
