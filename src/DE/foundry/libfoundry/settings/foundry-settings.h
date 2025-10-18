/* foundry-settings.h
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

#include "foundry-contextual.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SETTINGS       (foundry_settings_get_type())
#define FOUNDRY_TYPE_SETTINGS_LAYER (foundry_settings_layer_get_type())


/**
 * FoundrySettingsLayer:
 * @FOUNDRY_SETTINGS_LAYER_APPLICATION: Application-wide settings global to
 *   any new project opened or created with Foundry.
 * @FOUNDRY_SETTINGS_LAYER_PROJECT: Project-level overrides which take
 *   priority over %FOUNDRY_SETTINGS_LAYER_APPLICATION.
 * @FOUNDRY_SETTINGS_LAYER_USER: User-level overrides which take priority
 *   over %FOUNDRY_SETTINGS_LAYER_APPLICATION and
 *   %FOUNDRY_SETTINGS_LAYER_PROJECT.
 */
typedef enum _FoundrySettingsLayer
{
  FOUNDRY_SETTINGS_LAYER_APPLICATION,
  FOUNDRY_SETTINGS_LAYER_PROJECT,
  FOUNDRY_SETTINGS_LAYER_USER,
} FoundrySettingsLayer;

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundrySettings, foundry_settings, FOUNDRY, SETTINGS, FoundryContextual)

FOUNDRY_AVAILABLE_IN_ALL
GType            foundry_settings_layer_get_type              (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
FoundrySettings *foundry_settings_new                         (FoundryContext          *context,
                                                               const char              *schema_id);
FOUNDRY_AVAILABLE_IN_ALL
FoundrySettings *foundry_settings_new_with_path               (FoundryContext          *context,
                                                               const char              *schema_id,
                                                               const char              *path);
FOUNDRY_AVAILABLE_IN_ALL
GSettings       *foundry_settings_dup_layer                   (FoundrySettings         *self,
                                                               FoundrySettingsLayer     layer);
FOUNDRY_AVAILABLE_IN_ALL
const char      *foundry_settings_get_schema_id               (FoundrySettings         *self);
FOUNDRY_AVAILABLE_IN_ALL
GVariant        *foundry_settings_get_default_value           (FoundrySettings         *self,
                                                               const char              *key);
FOUNDRY_AVAILABLE_IN_ALL
GVariant        *foundry_settings_get_user_value              (FoundrySettings         *self,
                                                               const char              *key);
FOUNDRY_AVAILABLE_IN_ALL
GVariant        *foundry_settings_get_value                   (FoundrySettings         *self,
                                                               const char              *key);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_settings_set_value                   (FoundrySettings         *self,
                                                               const char              *key,
                                                               GVariant                *value);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_settings_get_boolean                 (FoundrySettings         *self,
                                                               const char              *key);
FOUNDRY_AVAILABLE_IN_ALL
double           foundry_settings_get_double                  (FoundrySettings         *self,
                                                               const char              *key);
FOUNDRY_AVAILABLE_IN_ALL
int              foundry_settings_get_int                     (FoundrySettings         *self,
                                                               const char              *key);
FOUNDRY_AVAILABLE_IN_ALL
char            *foundry_settings_get_string                  (FoundrySettings         *self,
                                                               const char              *key);
FOUNDRY_AVAILABLE_IN_ALL
guint            foundry_settings_get_uint                    (FoundrySettings         *self,
                                                               const char              *key);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_settings_set_boolean                 (FoundrySettings         *self,
                                                               const char              *key,
                                                               gboolean                 val);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_settings_set_double                  (FoundrySettings         *self,
                                                               const char              *key,
                                                               double                   val);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_settings_set_int                     (FoundrySettings         *self,
                                                               const char              *key,
                                                               int                      val);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_settings_set_string                  (FoundrySettings         *self,
                                                               const char              *key,
                                                               const char              *val);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_settings_set_uint                    (FoundrySettings         *self,
                                                               const char              *key,
                                                               guint                    val);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_settings_bind                        (FoundrySettings         *self,
                                                               const char              *key,
                                                               gpointer                 object,
                                                               const char              *property,
                                                               GSettingsBindFlags       flags);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_settings_bind_with_mapping           (FoundrySettings         *self,
                                                               const char              *key,
                                                               gpointer                 object,
                                                               const char              *property,
                                                               GSettingsBindFlags       flags,
                                                               GSettingsBindGetMapping  get_mapping,
                                                               GSettingsBindSetMapping  set_mapping,
                                                               gpointer                 user_data,
                                                               GDestroyNotify           destroy);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_settings_unbind                      (FoundrySettings         *self,
                                                               const char              *property);

G_END_DECLS
