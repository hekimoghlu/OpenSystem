/* spelling-cursor-private.h
 *
 * Copyright 2021-2023 Christian Hergert <chergert@redhat.com>
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

#include <gtk/gtk.h>

G_BEGIN_DECLS

typedef struct _SpellingCursor SpellingCursor;
typedef struct _CjhTextRegion  CjhTextRegion;

SpellingCursor *spelling_cursor_new               (GtkTextBuffer  *buffer,
                                                   CjhTextRegion  *region,
                                                   GtkTextTag     *no_spell_check_tag,
                                                   const char     *extra_word_chars);
void            spelling_cursor_free              (SpellingCursor *cursor);
gboolean        spelling_cursor_next              (SpellingCursor *cursor,
                                                   GtkTextIter    *word_begin,
                                                   GtkTextIter    *word_end);
gboolean        spelling_iter_forward_word_end    (GtkTextIter    *iter,
                                                   const char     *extra_word_chars);
gboolean        spelling_iter_backward_word_start (GtkTextIter    *iter,
                                                   const char     *extra_word_chars);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (SpellingCursor, spelling_cursor_free)

G_END_DECLS
