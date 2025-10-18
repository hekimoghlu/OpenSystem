/* foundry-search-dialog-private.h
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

#include <adwaita.h>
#include <foundry.h>

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SEARCH_DIALOG (foundry_search_dialog_get_type())

FOUNDRY_AVAILABLE_IN_1_1
G_DECLARE_FINAL_TYPE (FoundrySearchDialog, foundry_search_dialog, FOUNDRY, SEARCH_DIALOG, AdwDialog)

FOUNDRY_AVAILABLE_IN_1_1
GtkWidget      *foundry_search_dialog_new         (void);
FOUNDRY_AVAILABLE_IN_1_1
FoundryContext *foundry_search_dialog_dup_context (FoundrySearchDialog *self);
FOUNDRY_AVAILABLE_IN_1_1
void            foundry_search_dialog_set_context (FoundrySearchDialog *self,
                                                   FoundryContext      *context);

G_END_DECLS
