/*
 * foundry-panel.h
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <gtk/gtk.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_PANEL (foundry_panel_get_type())

FOUNDRY_AVAILABLE_IN_1_1
G_DECLARE_DERIVABLE_TYPE (FoundryPanel, foundry_panel, FOUNDRY, PANEL, GtkWidget)

struct _FoundryPanelClass
{
  GtkWidgetClass parent_class;

  /*< private >*/
  gpointer _reserved[13];
};

FOUNDRY_AVAILABLE_IN_1_1
FoundryPanel *foundry_panel_new           (const char   *id);
FOUNDRY_AVAILABLE_IN_1_1
const char   *foundry_panel_get_id        (FoundryPanel *self);
FOUNDRY_AVAILABLE_IN_1_1
const char   *foundry_panel_get_title     (FoundryPanel *self);
FOUNDRY_AVAILABLE_IN_1_1
void          foundry_panel_set_title     (FoundryPanel *self,
                                           const char   *title);
FOUNDRY_AVAILABLE_IN_1_1
GIcon        *foundry_panel_get_icon      (FoundryPanel *self);
FOUNDRY_AVAILABLE_IN_1_1
void          foundry_panel_set_icon      (FoundryPanel *self,
                                           GIcon        *icon);
FOUNDRY_AVAILABLE_IN_1_1
void          foundry_panel_set_icon_name (FoundryPanel *self,
                                           const char   *icon_name);
FOUNDRY_AVAILABLE_IN_1_1
GtkWidget    *foundry_panel_get_child     (FoundryPanel *self);
FOUNDRY_AVAILABLE_IN_1_1
void          foundry_panel_set_child     (FoundryPanel *self,
                                           GtkWidget    *child);

G_END_DECLS
