/* foundry-extension-set.h
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include <libpeas.h>

#include "foundry-contextual.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_EXTENSION_SET (foundry_extension_set_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryExtensionSet, foundry_extension_set, FOUNDRY, EXTENSION_SET, FoundryContextual)

typedef void (*FoundryExtensionSetForeachFunc) (FoundryExtensionSet *set,
                                                PeasPluginInfo      *plugin_info,
                                                GObject             *extension,
                                                gpointer             user_data);

FOUNDRY_AVAILABLE_IN_ALL
FoundryExtensionSet *foundry_extension_set_new                 (FoundryContext                 *context,
                                                                PeasEngine                     *engine,
                                                                GType                           interface_type,
                                                                const char                     *key,
                                                                const char                     *value,
                                                                ...) G_GNUC_NULL_TERMINATED;
FOUNDRY_AVAILABLE_IN_ALL
PeasEngine          *foundry_extension_set_get_engine          (FoundryExtensionSet            *self);
FOUNDRY_AVAILABLE_IN_ALL
GType                foundry_extension_set_get_interface_type  (FoundryExtensionSet            *self);
FOUNDRY_AVAILABLE_IN_ALL
const gchar         *foundry_extension_set_get_key             (FoundryExtensionSet            *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_extension_set_set_key             (FoundryExtensionSet            *self,
                                                                const gchar                    *key);
FOUNDRY_AVAILABLE_IN_ALL
const gchar         *foundry_extension_set_get_value           (FoundryExtensionSet            *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_extension_set_set_value           (FoundryExtensionSet            *self,
                                                                const gchar                    *value);
FOUNDRY_AVAILABLE_IN_ALL
guint                foundry_extension_set_get_n_extensions    (FoundryExtensionSet            *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_extension_set_foreach             (FoundryExtensionSet            *self,
                                                                FoundryExtensionSetForeachFunc  foreach_func,
                                                                gpointer                        user_data);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_extension_set_foreach_by_priority (FoundryExtensionSet            *self,
                                                                FoundryExtensionSetForeachFunc  foreach_func,
                                                                gpointer                        user_data);
FOUNDRY_AVAILABLE_IN_ALL
GObject             *foundry_extension_set_get_extension       (FoundryExtensionSet            *self,
                                                                PeasPluginInfo                 *plugin_info);

G_END_DECLS
