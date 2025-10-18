/* foundry-text-iter.h
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

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

typedef struct _FoundryTextIterVTable
{
  gboolean (*backward_char)           (FoundryTextIter       *iter);
  gboolean (*ends_line)               (const FoundryTextIter *iter);
  gboolean (*forward_char)            (FoundryTextIter       *iter);
  gboolean (*forward_line)            (FoundryTextIter       *iter);
  gunichar (*get_char)                (const FoundryTextIter *iter);
  gsize    (*get_line)                (const FoundryTextIter *iter);
  gsize    (*get_line_offset)         (const FoundryTextIter *iter);
  gsize    (*get_offset)              (const FoundryTextIter *iter);
  gboolean (*is_end)                  (const FoundryTextIter *iter);
  gboolean (*is_start)                (const FoundryTextIter *iter);
  gboolean (*move_to_line_and_offset) (FoundryTextIter       *iter,
                                       gsize                  line,
                                       gsize                  line_offset);
  gboolean (*starts_line)             (const FoundryTextIter *iter);
} FoundryTextIterVTable;

struct _FoundryTextIter
{
  /*< private >*/
  FoundryTextBuffer           *buffer;
  const FoundryTextIterVTable *vtable;
  gpointer                     _reserved[14];
};

FOUNDRY_AVAILABLE_IN_ALL
void     foundry_text_iter_init                     (FoundryTextIter             *iter,
                                                     FoundryTextBuffer           *buffer,
                                                     const FoundryTextIterVTable *vtable);
FOUNDRY_AVAILABLE_IN_ALL
gsize    foundry_text_iter_get_offset               (const FoundryTextIter       *iter);
FOUNDRY_AVAILABLE_IN_ALL
gsize    foundry_text_iter_get_line                 (const FoundryTextIter       *iter);
FOUNDRY_AVAILABLE_IN_ALL
gsize    foundry_text_iter_get_line_offset          (const FoundryTextIter       *iter);
FOUNDRY_AVAILABLE_IN_ALL
gboolean foundry_text_iter_ends_line                (const FoundryTextIter       *iter);
FOUNDRY_AVAILABLE_IN_ALL
gboolean foundry_text_iter_forward_char             (FoundryTextIter             *iter);
FOUNDRY_AVAILABLE_IN_ALL
gboolean foundry_text_iter_forward_line             (FoundryTextIter             *iter);
FOUNDRY_AVAILABLE_IN_ALL
gboolean foundry_text_iter_backward_char            (FoundryTextIter             *iter);
FOUNDRY_AVAILABLE_IN_ALL
gboolean foundry_text_iter_is_start                 (const FoundryTextIter       *iter);
FOUNDRY_AVAILABLE_IN_ALL
gboolean foundry_text_iter_is_end                   (const FoundryTextIter       *iter);
FOUNDRY_AVAILABLE_IN_ALL
gboolean foundry_text_iter_move_to_line_and_offset  (FoundryTextIter             *iter,
                                                     gsize                        line,
                                                     gsize                        line_offset);
FOUNDRY_AVAILABLE_IN_ALL
gboolean foundry_text_iter_starts_line              (const FoundryTextIter       *iter);

G_END_DECLS
