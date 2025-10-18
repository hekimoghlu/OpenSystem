/* foundry-diagnostics-gutter-renderer.h
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

#include <foundry.h>
#include <gtksourceview/gtksource.h>

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DIAGNOSTICS_GUTTER_RENDERER (foundry_diagnostics_gutter_renderer_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryDiagnosticsGutterRenderer, foundry_diagnostics_gutter_renderer, FOUNDRY, DIAGNOSTICS_GUTTER_RENDERER, GtkSourceGutterRenderer)

FOUNDRY_AVAILABLE_IN_ALL
GtkSourceGutterRenderer  *foundry_diagnostics_gutter_renderer_new             (FoundryOnTypeDiagnostics         *diagnostics);
FOUNDRY_AVAILABLE_IN_ALL
FoundryOnTypeDiagnostics *foundry_diagnostics_gutter_renderer_dup_diagnostics (FoundryDiagnosticsGutterRenderer *self);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_diagnostics_gutter_renderer_set_diagnostics (FoundryDiagnosticsGutterRenderer *self,
                                                                               FoundryOnTypeDiagnostics         *diagnostics);

G_END_DECLS
