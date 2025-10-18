/* spelling-dictionary-internal.h
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
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

#include "spelling-dictionary.h"

G_BEGIN_DECLS

#define SPELLING_DICTIONARY_GET_CLASS(obj) G_TYPE_INSTANCE_GET_CLASS(obj, SPELLING_TYPE_DICTIONARY, SpellingDictionaryClass)

typedef struct _SpellingBoundary
{
  guint offset;
  guint length;
  guint byte_offset;
  guint byte_length;
} SpellingBoundary;

struct _SpellingDictionary
{
  GObject parent_instance;
  const char *code;
  GMutex mutex;
};

struct _SpellingDictionaryClass
{
  GObjectClass parent_class;

  void         (*lock)                 (SpellingDictionary *self);
  void         (*unlock)               (SpellingDictionary *self);
  gboolean     (*contains_word)        (SpellingDictionary *self,
                                        const char         *word,
                                        gssize              word_len);
  char       **(*list_corrections)     (SpellingDictionary *self,
                                        const char         *word,
                                        gssize              word_len);
  void         (*add_word)             (SpellingDictionary *self,
                                        const char         *word);
  void         (*ignore_word)          (SpellingDictionary *self,
                                        const char         *word);
  const char  *(*get_extra_word_chars) (SpellingDictionary *self);
};

GtkBitset *_spelling_dictionary_check_words (SpellingDictionary     *self,
                                             const char             *text,
                                             const SpellingBoundary *positions,
                                             guint                   n_positions);

G_END_DECLS
