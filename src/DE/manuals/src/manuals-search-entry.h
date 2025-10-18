/* manuals-search-entry.h
 *
 * Copyright 2021-2024 Christian Hergert <chergert@redhat.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <gtk/gtk.h>

G_BEGIN_DECLS

#define MANUALS_TYPE_SEARCH_ENTRY (manuals_search_entry_get_type())

G_DECLARE_FINAL_TYPE (ManualsSearchEntry, manuals_search_entry, MANUALS, SEARCH_ENTRY, GtkWidget)

GtkWidget *manuals_search_entry_new                     (void);
guint      manuals_search_entry_get_occurrence_count    (ManualsSearchEntry *self);
void       manuals_search_entry_set_occurrence_count    (ManualsSearchEntry *self,
                                                         guint               occurrence_count);
guint      manuals_search_entry_get_occurrence_position (ManualsSearchEntry *self);
void       manuals_search_entry_set_occurrence_position (ManualsSearchEntry *self,
                                                         int                 occurrence_position);

G_END_DECLS
