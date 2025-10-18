/*
 * manuals-tab.h
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include <adwaita.h>
#include <foundry.h>

G_BEGIN_DECLS

#define MANUALS_TYPE_TAB (manuals_tab_get_type())

G_DECLARE_FINAL_TYPE (ManualsTab, manuals_tab, MANUALS, TAB, GtkWidget)

ManualsTab           *manuals_tab_new             (void);
ManualsTab           *manuals_tab_duplicate       (ManualsTab           *self);
GIcon                *manuals_tab_dup_icon        (ManualsTab           *self);
gboolean              manuals_tab_get_loading     (ManualsTab           *self);
char                 *manuals_tab_dup_title       (ManualsTab           *self);
gboolean              manuals_tab_can_go_back     (ManualsTab           *self);
gboolean              manuals_tab_can_go_forward  (ManualsTab           *self);
void                  manuals_tab_go_back         (ManualsTab           *self);
void                  manuals_tab_go_forward      (ManualsTab           *self);
FoundryDocumentation *manuals_tab_get_navigatable (ManualsTab           *self);
void                  manuals_tab_set_navigatable (ManualsTab           *self,
                                                   FoundryDocumentation *documentation);
void                  manuals_tab_load_uri        (ManualsTab           *self,
                                                   const char           *uri);
void                  manuals_tab_focus_search    (ManualsTab           *self);

G_END_DECLS
