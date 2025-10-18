/* plugin-doap-person.h
 *
 * Copyright 2015-2025 Christian Hergert <chergert@redhat.com>
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

G_BEGIN_DECLS

#define PLUGIN_TYPE_DOAP_PERSON (plugin_doap_person_get_type())

G_DECLARE_FINAL_TYPE (PluginDoapPerson, plugin_doap_person, PLUGIN, DOAP_PERSON, GObject)

PluginDoapPerson *plugin_doap_person_new       (void);
const char       *plugin_doap_person_get_name  (PluginDoapPerson *self);
void              plugin_doap_person_set_name  (PluginDoapPerson *self,
                                                const char       *name);
const char       *plugin_doap_person_get_email (PluginDoapPerson *self);
void              plugin_doap_person_set_email (PluginDoapPerson *self,
                                                const char       *email);

G_END_DECLS
