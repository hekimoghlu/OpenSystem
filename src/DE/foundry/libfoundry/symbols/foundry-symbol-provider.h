/* foundry-symbol-provider.h
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

#include "foundry-contextual.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SYMBOL_PROVIDER (foundry_symbol_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundrySymbolProvider, foundry_symbol_provider, FOUNDRY, SYMBOL_PROVIDER, FoundryContextual)

struct _FoundrySymbolProviderClass
{
  FoundryContextualClass parent_class;

  DexFuture *(*list_symbols)   (FoundrySymbolProvider *self,
                                GFile                 *file,
                                GBytes                *contents);
  DexFuture *(*find_symbol_at) (FoundrySymbolProvider *self,
                                GFile                 *file,
                                GBytes                *contents,
                                guint                  line,
                                guint                  line_offset);

  /*< private >*/
  gpointer _reserved[14];
};

FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_symbol_provider_list_symbols   (FoundrySymbolProvider *self,
                                                   GFile                 *file,
                                                   GBytes                *contents);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_symbol_provider_find_symbol_at (FoundrySymbolProvider *self,
                                                   GFile                 *file,
                                                   GBytes                *contents,
                                                   guint                  line,
                                                   guint                  line_offset);

G_END_DECLS
