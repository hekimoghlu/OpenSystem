/* foundry-workspace-child-private.h
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

#include <libpanel.h>

G_BEGIN_DECLS

#define FOUNDRY_TYPE_WORKSPACE_CHILD      (foundry_workspace_child_get_type())
#define FOUNDRY_TYPE_WORKSPACE_CHILD_KIND (foundry_workspace_child_kind_get_type())

G_DECLARE_FINAL_TYPE (FoundryWorkspaceChild, foundry_workspace_child, FOUNDRY, WORKSPACE_CHILD, GObject)

typedef enum _FoundryWorkspaceChildKind
{
  FOUNDRY_WORKSPACE_CHILD_PAGE,
  FOUNDRY_WORKSPACE_CHILD_PANEL,
} FoundryWorkspaceChildKind;

typedef enum _FoundryWorkspaceLayout
{
  FOUNDRY_WORKSPACE_LAYOUT_NARROW,
  FOUNDRY_WORKSPACE_LAYOUT_WIDE,
} FoundryWorkspaceLayout;

GType                      foundry_workspace_child_kind_get_type     (void) G_GNUC_CONST;
GType                      foundry_workspace_layout_get_type         (void) G_GNUC_CONST;
FoundryWorkspaceChild     *foundry_workspace_child_new               (FoundryWorkspaceChildKind  kind,
                                                                      PanelArea                  area);
FoundryWorkspaceChildKind  foundry_workspace_child_get_kind          (FoundryWorkspaceChild     *self);
const char                *foundry_workspace_child_get_title         (FoundryWorkspaceChild     *self);
void                       foundry_workspace_child_set_title         (FoundryWorkspaceChild     *self,
                                                                      const char                *title);
const char                *foundry_workspace_child_get_subtitle      (FoundryWorkspaceChild     *self);
void                       foundry_workspace_child_set_subtitle      (FoundryWorkspaceChild     *self,
                                                                      const char                *subtitle);
GIcon                     *foundry_workspace_child_get_icon          (FoundryWorkspaceChild     *self);
void                       foundry_workspace_child_set_icon          (FoundryWorkspaceChild     *self,
                                                                      GIcon                     *icon);
GtkWidget                 *foundry_workspace_child_get_child         (FoundryWorkspaceChild     *self);
void                       foundry_workspace_child_set_child         (FoundryWorkspaceChild     *self,
                                                                      GtkWidget                 *child);
void                       foundry_workspace_child_set_layout        (FoundryWorkspaceChild     *self,
                                                                      FoundryWorkspaceLayout     layout);
GtkWidget                 *foundry_workspace_child_get_wide_widget   (FoundryWorkspaceChild     *self);
GtkWidget                 *foundry_workspace_child_get_narrow_widget (FoundryWorkspaceChild     *self);
gboolean                   foundry_workspace_child_get_modified      (FoundryWorkspaceChild     *self);
void                       foundry_workspace_child_set_modified      (FoundryWorkspaceChild     *self,
                                                                      gboolean                   modified);
PanelArea                  foundry_workspace_child_get_area          (FoundryWorkspaceChild     *self);

G_END_DECLS
