/*
 * Copyright © 2001 Havoc Pennington
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
 */

#ifndef TERMINAL_ACCELS_H
#define TERMINAL_ACCELS_H

#include <adwaita.h>

G_BEGIN_DECLS

void terminal_accels_init (GApplication *application,
                           GSettings *settings,
                           gboolean use_headerbar);

void terminal_accels_shutdown (void);

GSettings *terminal_accels_get_settings (void);

#ifdef TERMINAL_PREFERENCES
void terminal_accels_populate_preferences (AdwPreferencesPage *page);
#endif

G_END_DECLS

#endif /* TERMINAL_ACCELS_H */
