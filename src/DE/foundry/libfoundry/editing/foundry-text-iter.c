/* foundry-text-iter.c
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

#include "config.h"

#include "foundry-text-buffer.h"
#include "foundry-text-iter-private.h"

void
foundry_text_iter_init (FoundryTextIter             *iter,
                        FoundryTextBuffer           *buffer,
                        const FoundryTextIterVTable *vtable)
{
  g_return_if_fail (iter != NULL);
  g_return_if_fail (FOUNDRY_IS_TEXT_BUFFER (buffer));
  g_return_if_fail (vtable != NULL);

  iter->buffer = buffer;
  iter->vtable = vtable;
}

gsize
foundry_text_iter_get_offset (const FoundryTextIter *iter)
{
  return iter->vtable->get_offset (iter);
}

gsize
foundry_text_iter_get_line (const FoundryTextIter *iter)
{
  return iter->vtable->get_line (iter);
}

gsize
foundry_text_iter_get_line_offset (const FoundryTextIter *iter)
{
  return iter->vtable->get_line_offset (iter);
}

gboolean
foundry_text_iter_forward_char (FoundryTextIter *iter)
{
  return iter->vtable->forward_char (iter);
}

gboolean
foundry_text_iter_backward_char (FoundryTextIter *iter)
{
  return iter->vtable->backward_char (iter);
}

gboolean
foundry_text_iter_is_start (const FoundryTextIter *iter)
{
  return iter->vtable->is_start (iter);
}

gboolean
foundry_text_iter_is_end (const FoundryTextIter *iter)
{
  return iter->vtable->is_end (iter);
}

gboolean
foundry_text_iter_ends_line (const FoundryTextIter *iter)
{
  return iter->vtable->ends_line (iter);
}

gboolean
foundry_text_iter_starts_line (const FoundryTextIter *iter)
{
  return iter->vtable->starts_line (iter);
}

gboolean
foundry_text_iter_move_to_line_and_offset (FoundryTextIter *iter,
                                           gsize            line,
                                           gsize            line_offset)
{
  return iter->vtable->move_to_line_and_offset (iter, line, line_offset);
}
