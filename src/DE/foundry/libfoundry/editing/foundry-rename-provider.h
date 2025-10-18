/* foundry-rename-provider.h
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

#include "foundry-contextual.h"
#include "foundry-text-iter.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_RENAME_PROVIDER (foundry_rename_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryRenameProvider, foundry_rename_provider, FOUNDRY, RENAME_PROVIDER, FoundryContextual)

struct _FoundryRenameProviderClass
{
  FoundryContextualClass parent_class;

  DexFuture *(*rename) (FoundryRenameProvider *self,
                        const FoundryTextIter *iter,
                        const char            *new_name);

  /*< private >*/
  gpointer _reserved[15];
};

FOUNDRY_AVAILABLE_IN_ALL
FoundryTextBuffer   *foundry_rename_provider_dup_buffer   (FoundryRenameProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTextDocument *foundry_rename_provider_dup_document (FoundryRenameProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_rename_provider_rename       (FoundryRenameProvider *self,
                                                           const FoundryTextIter *iter,
                                                           const char            *new_name);

G_END_DECLS
