/* manuals-application.h
 *
 * Copyright 2025 Christian Hergert
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
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <adwaita.h>
#include <foundry.h>

G_BEGIN_DECLS

#define MANUALS_TYPE_APPLICATION    (manuals_application_get_type())
#define MANUALS_APPLICATION_DEFAULT (MANUALS_APPLICATION(g_application_get_default()))

G_DECLARE_FINAL_TYPE (ManualsApplication, manuals_application, MANUALS, APPLICATION, AdwApplication)

ManualsApplication *manuals_application_new                (const char         *application_id,
                                                            GApplicationFlags   flags);
DexFuture          *manuals_application_load_foundry       (ManualsApplication *self) G_GNUC_WARN_UNUSED_RESULT;
gboolean            manuals_application_get_import_active  (ManualsApplication *self);
gboolean            manuals_application_control_is_pressed (void);
void                manuals_application_reload_content     (ManualsApplication *self);

G_END_DECLS
