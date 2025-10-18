/* foundry-layered-settings-private.h
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

#include <gio/gio.h>

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LAYERED_SETTINGS (foundry_layered_settings_get_type())

G_DECLARE_FINAL_TYPE (FoundryLayeredSettings, foundry_layered_settings, FOUNDRY, LAYERED_SETTINGS, GObject)

FoundryLayeredSettings  *foundry_layered_settings_new               (const char              *schema_id,
                                                                     const char              *path);
GSettingsSchemaKey      *foundry_layered_settings_get_key           (FoundryLayeredSettings  *self,
                                                                     const char              *key);
char                   **foundry_layered_settings_list_keys         (FoundryLayeredSettings  *self);
GVariant                *foundry_layered_settings_get_default_value (FoundryLayeredSettings  *self,
                                                                     const char              *key);
GVariant                *foundry_layered_settings_get_user_value    (FoundryLayeredSettings  *self,
                                                                     const char              *key);
GVariant                *foundry_layered_settings_get_value         (FoundryLayeredSettings  *self,
                                                                     const char              *key);
void                     foundry_layered_settings_set_value         (FoundryLayeredSettings  *self,
                                                                     const char              *key,
                                                                     GVariant                *value);
gboolean                 foundry_layered_settings_get_boolean       (FoundryLayeredSettings  *self,
                                                                     const char              *key);
double                   foundry_layered_settings_get_double        (FoundryLayeredSettings  *self,
                                                                     const char              *key);
int                      foundry_layered_settings_get_int           (FoundryLayeredSettings  *self,
                                                                     const char              *key);
char                    *foundry_layered_settings_get_string        (FoundryLayeredSettings  *self,
                                                                     const char              *key);
guint                    foundry_layered_settings_get_uint          (FoundryLayeredSettings  *self,
                                                                     const char              *key);
void                     foundry_layered_settings_set_boolean       (FoundryLayeredSettings  *self,
                                                                     const char              *key,
                                                                     gboolean                 val);
void                     foundry_layered_settings_set_double        (FoundryLayeredSettings  *self,
                                                                     const char              *key,
                                                                     double                   val);
void                     foundry_layered_settings_set_int           (FoundryLayeredSettings  *self,
                                                                     const char              *key,
                                                                     int                      val);
void                     foundry_layered_settings_set_string        (FoundryLayeredSettings  *self,
                                                                     const char              *key,
                                                                     const char              *val);
void                     foundry_layered_settings_set_uint          (FoundryLayeredSettings  *self,
                                                                     const char              *key,
                                                                     guint                    val);
void                     foundry_layered_settings_append            (FoundryLayeredSettings  *self,
                                                                     GSettings               *settings);
void                     foundry_layered_settings_bind              (FoundryLayeredSettings  *self,
                                                                     const char              *key,
                                                                     gpointer                 object,
                                                                     const char              *property,
                                                                     GSettingsBindFlags       flags);
void                     foundry_layered_settings_bind_with_mapping (FoundryLayeredSettings  *self,
                                                                     const char              *key,
                                                                     gpointer                 object,
                                                                     const char              *property,
                                                                     GSettingsBindFlags       flags,
                                                                     GSettingsBindGetMapping  get_mapping,
                                                                     GSettingsBindSetMapping  set_mapping,
                                                                     gpointer                 user_data,
                                                                     GDestroyNotify           destroy);
void                     foundry_layered_settings_unbind            (FoundryLayeredSettings  *self,
                                                                     const char              *property);

G_END_DECLS
