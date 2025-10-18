/* foundry-symbol.h
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SYMBOL (foundry_symbol_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundrySymbol, foundry_symbol, FOUNDRY, SYMBOL, GObject)

struct _FoundrySymbolClass
{
  GObjectClass parent_class;

  char      *(*dup_name)      (FoundrySymbol *self);
  DexFuture *(*find_parent)   (FoundrySymbol *self);
  DexFuture *(*list_children) (FoundrySymbol *self);

  /*< private >*/
  gpointer _reserved[12];
};

FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_symbol_dup_name      (FoundrySymbol *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_symbol_find_parent   (FoundrySymbol *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_symbol_list_children (FoundrySymbol *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_symbol_list_to_root  (FoundrySymbol *self);

G_END_DECLS
