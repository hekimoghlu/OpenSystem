/* foundry-fuzzy-index-private.h
 *
 * Copyright 2015-2025 Christian Hergert <chergert@redhat.com>
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

#include <glib-object.h>

G_BEGIN_DECLS

#define FOUNDRY_TYPE_FUZZY_INDEX (foundry_fuzzy_index_get_type())

typedef struct _FoundryFuzzyIndex FoundryFuzzyIndex;

typedef struct
{
   const char *key;
   gpointer    value;
   float       score;
   guint       id;
} FoundryFuzzyIndexMatch;

GType              foundry_fuzzy_index_get_type           (void);
FoundryFuzzyIndex *foundry_fuzzy_index_new                (gboolean           case_sensitive);
FoundryFuzzyIndex *foundry_fuzzy_index_new_with_free_func (gboolean           case_sensitive,
                                                           GDestroyNotify     free_func);
void               foundry_fuzzy_index_set_free_func      (FoundryFuzzyIndex *fuzzy,
                                                           GDestroyNotify     free_func);
void               foundry_fuzzy_index_begin_bulk_insert  (FoundryFuzzyIndex *fuzzy);
void               foundry_fuzzy_index_end_bulk_insert    (FoundryFuzzyIndex *fuzzy);
gboolean           foundry_fuzzy_index_contains           (FoundryFuzzyIndex *fuzzy,
                                                           const char        *key);
void               foundry_fuzzy_index_insert             (FoundryFuzzyIndex *fuzzy,
                                                           const char        *key,
                                                           gpointer           value);
GArray            *foundry_fuzzy_index_match              (FoundryFuzzyIndex *fuzzy,
                                                           const char        *needle,
                                                           gsize              max_matches);
void               foundry_fuzzy_index_remove             (FoundryFuzzyIndex *fuzzy,
                                                           const char        *key);
FoundryFuzzyIndex *foundry_fuzzy_index_ref                (FoundryFuzzyIndex *fuzzy);
void               foundry_fuzzy_index_unref              (FoundryFuzzyIndex *fuzzy);
char              *foundry_fuzzy_highlight                (const char        *str,
                                                           const char        *query,
                                                           gboolean           case_sensitive);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (FoundryFuzzyIndex, foundry_fuzzy_index_unref)

G_END_DECLS
