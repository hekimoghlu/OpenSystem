/*
 * manuals-path-bar.h
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

#include <foundry.h>
#include <gtk/gtk.h>

G_BEGIN_DECLS

#define MANUALS_TYPE_PATH_BAR (manuals_path_bar_get_type())

G_DECLARE_FINAL_TYPE (ManualsPathBar, manuals_path_bar, MANUALS, PATH_BAR, GtkWidget)

ManualsPathBar       *manuals_path_bar_new              (void);
FoundryDocumentation *manuals_path_bar_get_navigatable  (ManualsPathBar       *self);
void                  manuals_path_bar_set_navigatable  (ManualsPathBar       *self,
                                                         FoundryDocumentation *navigatable);
void                  manuals_path_bar_inhibit_scroll   (ManualsPathBar       *self);
void                  manuals_path_bar_uninhibit_scroll (ManualsPathBar       *self);

G_END_DECLS
