/* foundry-text-formatter.h
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

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TEXT_FORMATTER (foundry_text_formatter_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryTextFormatter, foundry_text_formatter, FOUNDRY, TEXT_FORMATTER, FoundryContextual)

struct _FoundryTextFormatterClass
{
  FoundryContextualClass parent_class;

  gboolean   (*can_format_range) (FoundryTextFormatter  *self);
  DexFuture *(*format)           (FoundryTextFormatter  *self);
  DexFuture *(*format_range)     (FoundryTextFormatter  *self,
                                  const FoundryTextIter *begin,
                                  const FoundryTextIter *end);

  /*< private >*/
  gpointer _reserved[5];
};

FOUNDRY_AVAILABLE_IN_ALL
FoundryTextBuffer   *foundry_text_formatter_dup_buffer       (FoundryTextFormatter *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTextDocument *foundry_text_formatter_dup_document     (FoundryTextFormatter *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean             foundry_text_formatter_can_format_range (FoundryTextFormatter *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_text_formatter_format           (FoundryTextFormatter *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_text_formatter_format_range     (FoundryTextFormatter  *self,
                                                              const FoundryTextIter *begin,
                                                              const FoundryTextIter *end);

G_END_DECLS
