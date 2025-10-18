/* foundry-tree-expander.h
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

#include <gtk/gtk.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TREE_EXPANDER (foundry_tree_expander_get_type())

FOUNDRY_AVAILABLE_IN_1_1
G_DECLARE_FINAL_TYPE (FoundryTreeExpander, foundry_tree_expander, FOUNDRY, TREE_EXPANDER, GtkWidget)

FOUNDRY_AVAILABLE_IN_1_1
GtkWidget      *foundry_tree_expander_new                    (void);
FOUNDRY_AVAILABLE_IN_1_1
GMenuModel     *foundry_tree_expander_get_menu_model         (FoundryTreeExpander *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_tree_expander_set_menu_model         (FoundryTreeExpander *self,
                                                              GMenuModel          *menu_model);
FOUNDRY_AVAILABLE_IN_1_1
GIcon          *foundry_tree_expander_get_icon               (FoundryTreeExpander *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_tree_expander_set_icon               (FoundryTreeExpander *self,
                                                              GIcon               *icon);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_tree_expander_set_icon_name          (FoundryTreeExpander *self,
                                                              const char          *icon_name);
FOUNDRY_AVAILABLE_IN_1_1
GIcon          *foundry_tree_expander_get_expanded_icon      (FoundryTreeExpander *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_tree_expander_set_expanded_icon      (FoundryTreeExpander *self,
                                                              GIcon               *icon);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_tree_expander_set_expanded_icon_name (FoundryTreeExpander *self,
                                                              const char          *expanded_icon_name);
FOUNDRY_AVAILABLE_IN_1_1
const char     *foundry_tree_expander_get_title              (FoundryTreeExpander *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_tree_expander_set_title              (FoundryTreeExpander *self,
                                                              const char          *title);
FOUNDRY_AVAILABLE_IN_1_1
gboolean        foundry_tree_expander_get_ignored            (FoundryTreeExpander *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_tree_expander_set_ignored            (FoundryTreeExpander *self,
                                                              gboolean             ignored);
FOUNDRY_AVAILABLE_IN_1_1
GtkWidget      *foundry_tree_expander_get_suffix             (FoundryTreeExpander *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_tree_expander_set_suffix             (FoundryTreeExpander *self,
                                                              GtkWidget           *suffix);
FOUNDRY_AVAILABLE_IN_1_1
GtkTreeListRow *foundry_tree_expander_get_list_row           (FoundryTreeExpander *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_tree_expander_set_list_row           (FoundryTreeExpander *self,
                                                              GtkTreeListRow      *list_row);
FOUNDRY_AVAILABLE_IN_1_1
gpointer        foundry_tree_expander_get_item               (FoundryTreeExpander *self);
FOUNDRY_AVAILABLE_IN_1_1
gboolean        foundry_tree_expander_get_use_markup         (FoundryTreeExpander *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_tree_expander_set_use_markup         (FoundryTreeExpander *self,
                                                              gboolean             use_markup);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_tree_expander_show_popover           (FoundryTreeExpander *self,
                                                              GtkPopover          *popover);

G_END_DECLS
