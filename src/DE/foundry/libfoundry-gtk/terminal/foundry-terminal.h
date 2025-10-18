/* foundry-terminal.h
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

#include <vte/vte.h>

#include "foundry-terminal-palette.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TERMINAL (foundry_terminal_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryTerminal, foundry_terminal, FOUNDRY, TERMINAL, VteTerminal)

struct _FoundryTerminalClass
{
  VteTerminalClass parent_class;

  /*< private >*/
  gpointer _reserved[12];
};

FOUNDRY_AVAILABLE_IN_ALL
GtkWidget              *foundry_terminal_new               (void);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTerminalPalette *foundry_terminal_get_palette       (FoundryTerminal        *self);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_terminal_set_palette       (FoundryTerminal        *self,
                                                            FoundryTerminalPalette *palette);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture              *foundry_terminal_list_palette_sets (void) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture              *foundry_terminal_find_palette_set  (const char             *name);

G_END_DECLS
