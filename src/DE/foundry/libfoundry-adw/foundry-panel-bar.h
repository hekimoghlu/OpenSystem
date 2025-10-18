/* foundry-panel-bar.h
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <gtk/gtk.h>

#include "foundry-version-macros.h"
#include "foundry-workspace.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_PANEL_BAR (foundry_panel_bar_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryPanelBar, foundry_panel_bar, FOUNDRY, PANEL_BAR, GtkWidget)

FOUNDRY_AVAILABLE_IN_ALL
GtkWidget        *foundry_panel_bar_new             (void);
FOUNDRY_AVAILABLE_IN_ALL
FoundryWorkspace *foundry_panel_bar_get_workspace   (FoundryPanelBar  *self);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_panel_bar_set_workspace   (FoundryPanelBar  *self,
                                                     FoundryWorkspace *workspace);
FOUNDRY_AVAILABLE_IN_ALL
gboolean          foundry_panel_bar_get_show_bottom (FoundryPanelBar  *self);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_panel_bar_set_show_bottom (FoundryPanelBar  *self,
                                                     gboolean          bottom);
FOUNDRY_AVAILABLE_IN_ALL
gboolean          foundry_panel_bar_get_show_start  (FoundryPanelBar  *self);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_panel_bar_set_show_start  (FoundryPanelBar  *self,
                                                     gboolean          start);

G_END_DECLS
