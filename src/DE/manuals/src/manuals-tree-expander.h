/*
 * manuals-tree-expander.h
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
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

#define MANUALS_TYPE_TREE_EXPANDER (manuals_tree_expander_get_type())

G_DECLARE_FINAL_TYPE (ManualsTreeExpander, manuals_tree_expander, MANUALS, TREE_EXPANDER, GtkWidget)

GtkWidget      *manuals_tree_expander_new                    (void);
GMenuModel     *manuals_tree_expander_get_menu_model         (ManualsTreeExpander *self);
void            manuals_tree_expander_set_menu_model         (ManualsTreeExpander *self,
                                                              GMenuModel          *menu_model);
GIcon          *manuals_tree_expander_get_icon               (ManualsTreeExpander *self);
void            manuals_tree_expander_set_icon               (ManualsTreeExpander *self,
                                                              GIcon               *icon);
void            manuals_tree_expander_set_icon_name          (ManualsTreeExpander *self,
                                                              const char          *icon_name);
GIcon          *manuals_tree_expander_get_expanded_icon      (ManualsTreeExpander *self);
void            manuals_tree_expander_set_expanded_icon      (ManualsTreeExpander *self,
                                                              GIcon               *icon);
void            manuals_tree_expander_set_expanded_icon_name (ManualsTreeExpander *self,
                                                              const char          *expanded_icon_name);
const char     *manuals_tree_expander_get_title              (ManualsTreeExpander *self);
void            manuals_tree_expander_set_title              (ManualsTreeExpander *self,
                                                              const char          *title);
gboolean        manuals_tree_expander_get_ignored            (ManualsTreeExpander *self);
void            manuals_tree_expander_set_ignored            (ManualsTreeExpander *self,
                                                              gboolean             ignored);
GtkWidget      *manuals_tree_expander_get_suffix             (ManualsTreeExpander *self);
void            manuals_tree_expander_set_suffix             (ManualsTreeExpander *self,
                                                              GtkWidget           *suffix);
GtkTreeListRow *manuals_tree_expander_get_list_row           (ManualsTreeExpander *self);
void            manuals_tree_expander_set_list_row           (ManualsTreeExpander *self,
                                                              GtkTreeListRow      *list_row);
gpointer        manuals_tree_expander_get_item               (ManualsTreeExpander *self);
gboolean        manuals_tree_expander_get_use_markup         (ManualsTreeExpander *self);
void            manuals_tree_expander_set_use_markup         (ManualsTreeExpander *self,
                                                              gboolean             use_markup);
void            manuals_tree_expander_show_popover           (ManualsTreeExpander *self,
                                                              GtkPopover          *popover);

G_END_DECLS

