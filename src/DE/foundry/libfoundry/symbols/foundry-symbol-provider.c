/* foundry-symbol-provider.c
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

#include "config.h"

#include "foundry-symbol-provider.h"
#include "foundry-util.h"

G_DEFINE_ABSTRACT_TYPE (FoundrySymbolProvider, foundry_symbol_provider, FOUNDRY_TYPE_CONTEXTUAL)

static void
foundry_symbol_provider_class_init (FoundrySymbolProviderClass *klass)
{
}

static void
foundry_symbol_provider_init (FoundrySymbolProvider *self)
{
}

/**
 * foundry_symbol_provider_list_symbols:
 * @self: a [class@Foundry.SymbolProvider]
 * @file: a [iface@Gio.File]
 * @contents: (nullable): optional modified contents for the file
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.Symbol] or rejects with error
 */
DexFuture *
foundry_symbol_provider_list_symbols (FoundrySymbolProvider *self,
                                      GFile                 *file,
                                      GBytes                *contents)
{
  dex_return_error_if_fail (FOUNDRY_IS_SYMBOL_PROVIDER (self));
  dex_return_error_if_fail (G_IS_FILE (file));

  if (FOUNDRY_SYMBOL_PROVIDER_GET_CLASS (self)->list_symbols)
    return FOUNDRY_SYMBOL_PROVIDER_GET_CLASS (self)->list_symbols (self, file, contents);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_symbol_provider_find_symbol_at:
 * @self: a [class@Foundry.SymbolProvider]
 * @file: a [iface@Gio.File]
 * @contents: (nullable):
 * @line: the line number (starting from 0)
 * @line_offset: the character offset (starting from 0)
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.Symbol] or rejects with error
 */
DexFuture *
foundry_symbol_provider_find_symbol_at (FoundrySymbolProvider *self,
                                        GFile                 *file,
                                        GBytes                *contents,
                                        guint                  line,
                                        guint                  line_offset)
{
  dex_return_error_if_fail (FOUNDRY_IS_SYMBOL_PROVIDER (self));
  dex_return_error_if_fail (G_IS_FILE (file));

  if (FOUNDRY_SYMBOL_PROVIDER_GET_CLASS (self)->find_symbol_at)
    return FOUNDRY_SYMBOL_PROVIDER_GET_CLASS (self)->find_symbol_at (self, file, contents, line, line_offset);

  return foundry_future_new_not_supported ();
}
