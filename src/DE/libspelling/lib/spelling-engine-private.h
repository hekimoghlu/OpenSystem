/* spelling-engine-private.h
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

#include <gtk/gtk.h>

#include "spelling-dictionary-internal.h"

G_BEGIN_DECLS

#define SPELLING_TYPE_ENGINE (spelling_engine_get_type())

typedef struct _SpellingAdapter
{
  gboolean            (*check_enabled)               (gpointer   instance);
  guint               (*get_cursor)                  (gpointer   instance);
  char               *(*copy_text)                   (gpointer   instance,
                                                      guint      position,
                                                      guint      length);
  void                (*apply_tag)                   (gpointer   instance,
                                                      guint      position,
                                                      guint      length);
  void                (*clear_tag)                   (gpointer   instance,
                                                      guint      position,
                                                      guint      length);
  gboolean            (*backward_word_start)         (gpointer   instance,
                                                      guint     *position);
  gboolean            (*forward_word_end)            (gpointer   instance,
                                                      guint     *position);
  void                (*intersect_spellcheck_region) (gpointer   instance,
                                                      GtkBitset *region);
  PangoLanguage      *(*get_language)                (gpointer   instance);
  SpellingDictionary *(*get_dictionary)              (gpointer   instance);
} SpellingAdapter;

G_DECLARE_FINAL_TYPE (SpellingEngine, spelling_engine, SPELLING, ENGINE, GObject)

SpellingEngine *spelling_engine_new                 (const SpellingAdapter *adapter,
                                                     GObject               *instance);
void            spelling_engine_before_insert_text  (SpellingEngine        *self,
                                                     guint                  position,
                                                     guint                  length);
void            spelling_engine_after_insert_text   (SpellingEngine        *self,
                                                     guint                  position,
                                                     guint                  length);
void            spelling_engine_before_delete_range (SpellingEngine        *self,
                                                     guint                  position,
                                                     guint                  length);
void            spelling_engine_after_delete_range  (SpellingEngine        *self,
                                                     guint                  position);
void            spelling_engine_iteration           (SpellingEngine        *self);
void            spelling_engine_invalidate          (SpellingEngine        *self,
                                                     guint                  position,
                                                     guint                  length);
void            spelling_engine_invalidate_all      (SpellingEngine        *self);

G_END_DECLS
