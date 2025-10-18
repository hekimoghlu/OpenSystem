/* foundry-workspace.h
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

#include <foundry.h>

#include "foundry-page.h"
#include "foundry-panel.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_WORKSPACE (foundry_workspace_get_type())

FOUNDRY_AVAILABLE_IN_1_1
G_DECLARE_FINAL_TYPE (FoundryWorkspace, foundry_workspace, FOUNDRY, WORKSPACE, GtkWidget)

FOUNDRY_AVAILABLE_IN_1_1
GtkWidget      *foundry_workspace_new                    (void);
FOUNDRY_AVAILABLE_IN_1_1
FoundryContext *foundry_workspace_get_context            (FoundryWorkspace *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_set_context            (FoundryWorkspace *self,
                                                          FoundryContext   *context);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_add_sidebar_panel      (FoundryWorkspace *self,
                                                          FoundryPanel     *panel);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_add_bottom_panel       (FoundryWorkspace *self,
                                                          FoundryPanel     *panel);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_remove_panel           (FoundryWorkspace *self,
                                                          FoundryPanel     *panel);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_add_page               (FoundryWorkspace *self,
                                                          FoundryPage      *page);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_remove_page            (FoundryWorkspace *self,
                                                          FoundryPage      *page);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_foreach_page           (FoundryWorkspace *self,
                                                          GFunc             callback,
                                                          gpointer          user_data);
FOUNDRY_AVAILABLE_IN_1_1
GMenuModel     *foundry_workspace_get_primary_menu       (FoundryWorkspace *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_set_primary_menu       (FoundryWorkspace *self,
                                                          GMenuModel       *menu);
FOUNDRY_AVAILABLE_IN_1_1
GtkWidget      *foundry_workspace_get_titlebar           (FoundryWorkspace *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_set_titlebar           (FoundryWorkspace *self,
                                                          GtkWidget        *titlebar);
FOUNDRY_AVAILABLE_IN_1_1
GtkWidget      *foundry_workspace_get_sidebar_titlebar   (FoundryWorkspace *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_set_sidebar_titlebar   (FoundryWorkspace *self,
                                                          GtkWidget        *sidebar_titlebar);
FOUNDRY_AVAILABLE_IN_1_1
GtkWidget      *foundry_workspace_get_collapsed_titlebar (FoundryWorkspace *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_set_collapsed_titlebar (FoundryWorkspace *self,
                                                          GtkWidget        *collapsed_titlebar);
FOUNDRY_AVAILABLE_IN_1_1
GtkWidget      *foundry_workspace_get_status_widget      (FoundryWorkspace *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_set_status_widget      (FoundryWorkspace *self,
                                                          GtkWidget        *status_widget);
FOUNDRY_AVAILABLE_IN_1_1
FoundryPage    *foundry_workspace_get_active_page        (FoundryWorkspace *self);
FOUNDRY_AVAILABLE_IN_1_1
gboolean        foundry_workspace_get_collapsed          (FoundryWorkspace *self);
FOUNDRY_AVAILABLE_IN_1_1
gboolean        foundry_workspace_get_show_sidebar       (FoundryWorkspace *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_set_show_sidebar       (FoundryWorkspace *self,
                                                          gboolean          show_sidebar);
FOUNDRY_AVAILABLE_IN_1_1
gboolean        foundry_workspace_get_show_auxillary     (FoundryWorkspace *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_set_show_auxillary     (FoundryWorkspace *self,
                                                          gboolean          show_auxillary);
FOUNDRY_AVAILABLE_IN_1_1
gboolean        foundry_workspace_get_show_utilities     (FoundryWorkspace *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_workspace_set_show_utilities     (FoundryWorkspace *self,
                                                          gboolean          show_utilities);

G_END_DECLS
