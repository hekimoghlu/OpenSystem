/* foundry-menu-manager.h
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

#include <gio/gio.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_MENU_MANAGER (foundry_menu_manager_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryMenuManager, foundry_menu_manager, FOUNDRY, MENU_MANAGER, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryMenuManager *foundry_menu_manager_get_default          (void);
FOUNDRY_AVAILABLE_IN_ALL
FoundryMenuManager *foundry_menu_manager_new                  (void);
FOUNDRY_AVAILABLE_IN_ALL
guint               foundry_menu_manager_add_filename         (FoundryMenuManager  *self,
                                                               const char          *filename,
                                                               GError             **error);
FOUNDRY_AVAILABLE_IN_ALL
guint               foundry_menu_manager_add_resource         (FoundryMenuManager  *self,
                                                               const char          *resource,
                                                               GError             **error);
FOUNDRY_AVAILABLE_IN_ALL
guint               foundry_menu_manager_merge                (FoundryMenuManager  *self,
                                                               const char          *menu_id,
                                                               GMenuModel          *menu_model);
FOUNDRY_AVAILABLE_IN_ALL
void                foundry_menu_manager_remove               (FoundryMenuManager  *self,
                                                               guint                merge_id);
FOUNDRY_AVAILABLE_IN_ALL
GMenu              *foundry_menu_manager_get_menu_by_id       (FoundryMenuManager  *self,
                                                               const char          *menu_id);
FOUNDRY_AVAILABLE_IN_ALL
const char * const *foundry_menu_manager_get_menu_ids         (FoundryMenuManager  *self);
FOUNDRY_AVAILABLE_IN_ALL
void                foundry_menu_manager_set_attribute_string (FoundryMenuManager  *self,
                                                               GMenu               *menu,
                                                               guint                position,
                                                               const char          *attribute,
                                                               const char          *value);
FOUNDRY_AVAILABLE_IN_ALL
GMenu              *foundry_menu_manager_find_item_by_id      (FoundryMenuManager  *self,
                                                               const char          *id,
                                                               guint               *position);

G_END_DECLS
