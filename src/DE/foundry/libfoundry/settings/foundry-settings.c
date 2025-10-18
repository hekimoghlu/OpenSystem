/* foundry-settings.c
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

#include "config.h"

#include <glib/gi18n-lib.h>

#include <stdlib.h>

#define G_SETTINGS_ENABLE_BACKEND
#include <gio/gsettingsbackend.h>

#include "foundry-context-private.h"
#include "foundry-debug.h"
#include "foundry-marshal.h"
#include "foundry-settings.h"
#include "foundry-layered-settings-private.h"
#include "foundry-util-private.h"

struct _FoundrySettings
{
  FoundryContextual       parent_instance;
  FoundryLayeredSettings *layered_settings;
  char                   *schema_id;
  char                   *path;
  GSettings              *app_settings;
  GSettings              *project_settings;
  GSettings              *user_settings;
};

static void action_group_iface_init (GActionGroupInterface *iface);

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundrySettings, foundry_settings, FOUNDRY_TYPE_CONTEXTUAL,
                               G_IMPLEMENT_INTERFACE (G_TYPE_ACTION_GROUP, action_group_iface_init))

G_DEFINE_ENUM_TYPE (FoundrySettingsLayer, foundry_settings_layer,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_SETTINGS_LAYER_APPLICATION, "application"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_SETTINGS_LAYER_PROJECT, "project"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_SETTINGS_LAYER_USER, "user"))

enum {
  PROP_0,
  PROP_PATH,
  PROP_SCHEMA_ID,
  N_PROPS
};

enum {
  CHANGED,
  N_SIGNALS
};

static GParamSpec *properties [N_PROPS];
static guint signals [N_SIGNALS];

static const GVariantType *
_g_variant_type_intern (const GVariantType *type)
{
  g_autofree char *str = NULL;

  if (type == NULL)
    return NULL;

  str = g_variant_type_dup_string (type);
  return G_VARIANT_TYPE (g_intern_string (str));
}

static void
foundry_settings_set_schema_id (FoundrySettings *self,
                                const char      *schema_id)
{
  g_assert (FOUNDRY_IS_SETTINGS (self));
  g_assert (schema_id != NULL);

  if (g_set_str (&self->schema_id, schema_id))
    g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_SCHEMA_ID]);
}

static void
foundry_settings_layered_settings_changed_cb (FoundrySettings        *self,
                                              const char             *key,
                                              FoundryLayeredSettings *layered_settings)
{
  g_autoptr(GVariant) value = NULL;

  g_assert (key != NULL);
  g_assert (FOUNDRY_IS_LAYERED_SETTINGS (layered_settings));

  g_signal_emit (self, signals[CHANGED], g_quark_from_string (key), key);

  value = foundry_layered_settings_get_value (self->layered_settings, key);
  g_action_group_action_state_changed (G_ACTION_GROUP (self), key, value);
}

static void
foundry_settings_constructed (GObject *object)
{
  FoundrySettings *self = (FoundrySettings *)object;
  g_autoptr(GSettingsBackend) project_backend = NULL;
  g_autoptr(GSettingsBackend) user_backend = NULL;
  g_autoptr(GSettingsSchema) schema = NULL;
  g_autoptr(FoundryContext) context = NULL;

  FOUNDRY_ENTRY;

  G_OBJECT_CLASS (foundry_settings_parent_class)->constructed (object);

  if (!(context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    {
      g_error ("Attempt to create a FoundrySettings without a context!");
      return;
    }

  if (self->schema_id == NULL)
    {
      g_error ("You must set %s:schema-id during construction",
               G_OBJECT_TYPE_NAME (self));
      return;
    }

  if (self->path == NULL)
    {
      GSettingsSchemaSource *source = g_settings_schema_source_get_default ();

      if (!(schema = g_settings_schema_source_lookup (source, self->schema_id, TRUE)))
        {
          g_error ("Failed to locate schema %s", self->schema_id);
          return;
        }

      self->path = g_strdup (g_settings_schema_get_path (schema));
    }

  self->layered_settings = foundry_layered_settings_new (self->schema_id, self->path);
  g_signal_connect_object (self->layered_settings,
                           "changed",
                           G_CALLBACK (foundry_settings_layered_settings_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  user_backend = _foundry_context_dup_user_settings_backend (context);
  self->user_settings = g_settings_new_with_backend_and_path (self->schema_id, user_backend, self->path);
  foundry_layered_settings_append (self->layered_settings, self->user_settings);

  project_backend = _foundry_context_dup_project_settings_backend (context);
  self->project_settings = g_settings_new_with_backend_and_path (self->schema_id, project_backend, self->path);
  foundry_layered_settings_append (self->layered_settings, self->project_settings);

  self->app_settings = g_settings_new_with_path (self->schema_id, self->path);
  foundry_layered_settings_append (self->layered_settings, self->app_settings);

  FOUNDRY_EXIT;
}

static void
foundry_settings_finalize (GObject *object)
{
  FoundrySettings *self = (FoundrySettings *)object;

  g_clear_object (&self->layered_settings);
  g_clear_object (&self->app_settings);
  g_clear_object (&self->project_settings);
  g_clear_object (&self->user_settings);

  g_clear_pointer (&self->schema_id, g_free);
  g_clear_pointer (&self->path, g_free);

  G_OBJECT_CLASS (foundry_settings_parent_class)->finalize (object);
}

static void
foundry_settings_get_property (GObject    *object,
                               guint       prop_id,
                               GValue     *value,
                               GParamSpec *pspec)
{
  FoundrySettings *self = FOUNDRY_SETTINGS (object);

  switch (prop_id)
    {
    case PROP_PATH:
      g_value_set_string (value, self->path);
      break;

    case PROP_SCHEMA_ID:
      g_value_set_string (value, foundry_settings_get_schema_id (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_settings_set_property (GObject      *object,
                           guint         prop_id,
                           const GValue *value,
                           GParamSpec   *pspec)
{
  FoundrySettings *self = FOUNDRY_SETTINGS (object);

  switch (prop_id)
    {
    case PROP_PATH:
      self->path = g_value_dup_string (value);
      break;

    case PROP_SCHEMA_ID:
      foundry_settings_set_schema_id (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_settings_class_init (FoundrySettingsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->constructed = foundry_settings_constructed;
  object_class->finalize = foundry_settings_finalize;
  object_class->get_property = foundry_settings_get_property;
  object_class->set_property = foundry_settings_set_property;

  properties[PROP_PATH] =
    g_param_spec_string ("path",
                         "Path",
                         "The path to use for for app settings",
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SCHEMA_ID] =
    g_param_spec_string ("schema-id",
                         "Schema ID",
                         "Schema ID",
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  signals[CHANGED] =
    g_signal_new ("changed",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST | G_SIGNAL_DETAILED,
                  0,
                  NULL, NULL,
                  foundry_marshal_VOID__STRING,
                  G_TYPE_NONE,
                  1,
                  G_TYPE_STRING | G_SIGNAL_TYPE_STATIC_SCOPE);
  g_signal_set_va_marshaller (signals [CHANGED],
                              G_TYPE_FROM_CLASS (klass),
                              foundry_marshal_VOID__STRINGv);
}

static void
foundry_settings_init (FoundrySettings *self)
{
}

FoundrySettings *
foundry_settings_new (FoundryContext *context,
                      const char     *schema_id)
{
  FoundrySettings *ret;

  FOUNDRY_ENTRY;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (schema_id != NULL, NULL);

  ret = g_object_new (FOUNDRY_TYPE_SETTINGS,
                      "context", context,
                      "schema-id", schema_id,
                      NULL);

  FOUNDRY_RETURN (ret);
}

FoundrySettings *
foundry_settings_new_with_path (FoundryContext *context,
                                const char     *schema_id,
                                const char     *path)
{
  FoundrySettings *ret;

  FOUNDRY_ENTRY;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (schema_id != NULL, NULL);

  ret = g_object_new (FOUNDRY_TYPE_SETTINGS,
                      "context", context,
                      "schema-id", schema_id,
                      "path", path,
                      NULL);

  FOUNDRY_RETURN (ret);
}

const char *
foundry_settings_get_schema_id (FoundrySettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SETTINGS (self), NULL);

  return self->schema_id;
}

GVariant *
foundry_settings_get_default_value (FoundrySettings *self,
                                    const char      *key)
{
  g_return_val_if_fail (FOUNDRY_IS_SETTINGS (self), NULL);
  g_return_val_if_fail (key != NULL, NULL);

  return foundry_layered_settings_get_default_value (self->layered_settings, key);
}

GVariant *
foundry_settings_get_user_value (FoundrySettings *self,
                                 const char      *key)
{
  g_return_val_if_fail (FOUNDRY_IS_SETTINGS (self), NULL);
  g_return_val_if_fail (key != NULL, NULL);

  return foundry_layered_settings_get_user_value (self->layered_settings, key);
}

GVariant *
foundry_settings_get_value (FoundrySettings *self,
                            const char      *key)
{
  g_return_val_if_fail (FOUNDRY_IS_SETTINGS (self), NULL);
  g_return_val_if_fail (key != NULL, NULL);

  return foundry_layered_settings_get_value (self->layered_settings, key);
}

void
foundry_settings_set_value (FoundrySettings *self,
                            const char      *key,
                            GVariant        *value)
{
  g_return_if_fail (FOUNDRY_IS_SETTINGS (self));
  g_return_if_fail (key != NULL);

  return foundry_layered_settings_set_value (self->layered_settings, key, value);
}

gboolean
foundry_settings_get_boolean (FoundrySettings *self,
                              const char      *key)
{
  g_return_val_if_fail (FOUNDRY_IS_SETTINGS (self), FALSE);
  g_return_val_if_fail (key != NULL, FALSE);

  return foundry_layered_settings_get_boolean (self->layered_settings, key);
}

double
foundry_settings_get_double (FoundrySettings *self,
                             const char      *key)
{
  g_return_val_if_fail (FOUNDRY_IS_SETTINGS (self), 0.0);
  g_return_val_if_fail (key != NULL, 0.0);

  return foundry_layered_settings_get_double (self->layered_settings, key);
}

int
foundry_settings_get_int (FoundrySettings *self,
                          const char      *key)
{
  g_return_val_if_fail (FOUNDRY_IS_SETTINGS (self), 0);
  g_return_val_if_fail (key != NULL, 0);

  return foundry_layered_settings_get_int (self->layered_settings, key);
}

char *
foundry_settings_get_string (FoundrySettings *self,
                             const char      *key)
{
  g_return_val_if_fail (FOUNDRY_IS_SETTINGS (self), NULL);
  g_return_val_if_fail (key != NULL, NULL);

  return foundry_layered_settings_get_string (self->layered_settings, key);
}

guint
foundry_settings_get_uint (FoundrySettings *self,
                           const char      *key)
{
  g_return_val_if_fail (FOUNDRY_IS_SETTINGS (self), 0);
  g_return_val_if_fail (key != NULL, 0);

  return foundry_layered_settings_get_uint (self->layered_settings, key);
}

void
foundry_settings_set_boolean (FoundrySettings *self,
                              const char      *key,
                              gboolean         val)
{
  g_return_if_fail (FOUNDRY_IS_SETTINGS (self));
  g_return_if_fail (key != NULL);

  foundry_layered_settings_set_boolean (self->layered_settings, key, val);
}

void
foundry_settings_set_double (FoundrySettings *self,
                             const char      *key,
                             double           val)
{
  g_return_if_fail (FOUNDRY_IS_SETTINGS (self));
  g_return_if_fail (key != NULL);

  foundry_layered_settings_set_double (self->layered_settings, key, val);
}

void
foundry_settings_set_int (FoundrySettings *self,
                          const char      *key,
                          int              val)
{
  g_return_if_fail (FOUNDRY_IS_SETTINGS (self));
  g_return_if_fail (key != NULL);

  foundry_layered_settings_set_int (self->layered_settings, key, val);
}

void
foundry_settings_set_string (FoundrySettings *self,
                             const char      *key,
                             const char      *val)
{
  g_return_if_fail (FOUNDRY_IS_SETTINGS (self));
  g_return_if_fail (key != NULL);

  foundry_layered_settings_set_string (self->layered_settings, key, val);
}

void
foundry_settings_set_uint (FoundrySettings *self,
                           const char      *key,
                           guint            val)
{
  g_return_if_fail (FOUNDRY_IS_SETTINGS (self));
  g_return_if_fail (key != NULL);

  foundry_layered_settings_set_uint (self->layered_settings, key, val);
}

void
foundry_settings_bind (FoundrySettings        *self,
                       const char             *key,
                       gpointer                object,
                       const char             *property,
                       GSettingsBindFlags      flags)
{
  g_return_if_fail (FOUNDRY_IS_SETTINGS (self));
  g_return_if_fail (key != NULL);
  g_return_if_fail (G_IS_OBJECT (object));
  g_return_if_fail (property != NULL);

  foundry_layered_settings_bind (self->layered_settings, key, object, property, flags);
}

/**
 * foundry_settings_bind_with_mapping:
 * @self: An #FoundrySettings
 * @key: The settings key
 * @object: the object to bind to
 * @property: the property of @object to bind to
 * @flags: flags for the binding
 * @get_mapping: (allow-none) (scope notified): variant to value mapping
 * @set_mapping: (allow-none) (scope notified): value to variant mapping
 * @user_data: user data for @get_mapping and @set_mapping
 * @destroy: destroy function to cleanup @user_data.
 *
 * Like foundry_settings_bind() but allows transforming to and from settings storage using
 * @get_mapping and @set_mapping transformation functions.
 *
 * Call foundry_settings_unbind() to unbind the mapping.
 */
void
foundry_settings_bind_with_mapping (FoundrySettings         *self,
                                    const char              *key,
                                    gpointer                 object,
                                    const char              *property,
                                    GSettingsBindFlags       flags,
                                    GSettingsBindGetMapping  get_mapping,
                                    GSettingsBindSetMapping  set_mapping,
                                    gpointer                 user_data,
                                    GDestroyNotify           destroy)
{
  g_return_if_fail (FOUNDRY_IS_SETTINGS (self));
  g_return_if_fail (key != NULL);
  g_return_if_fail (G_IS_OBJECT (object));
  g_return_if_fail (property != NULL);

  foundry_layered_settings_bind_with_mapping (self->layered_settings, key, object, property, flags,
                                              get_mapping, set_mapping, user_data, destroy);
}

void
foundry_settings_unbind (FoundrySettings *self,
                         const char      *property)
{
  g_return_if_fail (FOUNDRY_IS_SETTINGS (self));
  g_return_if_fail (property != NULL);

  foundry_layered_settings_unbind (self->layered_settings, property);
}

static gboolean
foundry_settings_has_action (GActionGroup *group,
                             const char   *action_name)
{
  FoundrySettings *self = FOUNDRY_SETTINGS (group);
  g_auto(GStrv) keys = foundry_layered_settings_list_keys (self->layered_settings);

  return g_strv_contains ((const char * const *)keys, action_name);
}

static char **
foundry_settings_list_actions (GActionGroup *group)
{
  FoundrySettings *self = FOUNDRY_SETTINGS (group);
  return foundry_layered_settings_list_keys (self->layered_settings);
}

static gboolean
foundry_settings_get_action_enabled (GActionGroup *group,
                                     const char   *action_name)
{
  return TRUE;
}

static GVariant *
foundry_settings_get_action_state (GActionGroup *group,
                                   const char   *action_name)
{
  FoundrySettings *self = FOUNDRY_SETTINGS (group);

  return foundry_layered_settings_get_value (self->layered_settings, action_name);
}

static GVariant *
foundry_settings_get_action_state_hint (GActionGroup *group,
                                        const char   *action_name)
{
  FoundrySettings *self = FOUNDRY_SETTINGS (group);
  g_autoptr(GSettingsSchemaKey) key = foundry_layered_settings_get_key (self->layered_settings, action_name);
  return g_settings_schema_key_get_range (key);
}

static void
foundry_settings_change_action_state (GActionGroup *group,
                                      const char   *action_name,
                                      GVariant     *value)
{
  FoundrySettings *self = FOUNDRY_SETTINGS (group);
  g_autoptr(GSettingsSchemaKey) key = foundry_layered_settings_get_key (self->layered_settings, action_name);

  if (g_variant_is_of_type (value, g_settings_schema_key_get_value_type (key)) &&
      g_settings_schema_key_range_check (key, value))
    {
      g_autoptr(GVariant) hold = g_variant_ref_sink (value);

      foundry_layered_settings_set_value (self->layered_settings, action_name, hold);
      g_action_group_action_state_changed (group, action_name, hold);
    }
}

static const GVariantType *
foundry_settings_get_action_state_type (GActionGroup *group,
                                        const char   *action_name)
{
  FoundrySettings *self = FOUNDRY_SETTINGS (group);
  g_autoptr(GSettingsSchemaKey) key = foundry_layered_settings_get_key (self->layered_settings, action_name);
  const GVariantType *type = g_settings_schema_key_get_value_type (key);

  return _g_variant_type_intern (type);
}

static void
foundry_settings_activate_action (GActionGroup *group,
                                  const char   *action_name,
                                  GVariant     *parameter)
{
  FoundrySettings *self = FOUNDRY_SETTINGS (group);
  g_autoptr(GSettingsSchemaKey) key = foundry_layered_settings_get_key (self->layered_settings, action_name);
  g_autoptr(GVariant) default_value = g_settings_schema_key_get_default_value (key);

  if (g_variant_is_of_type (default_value, G_VARIANT_TYPE_BOOLEAN))
    {
      GVariant *old;

      if (parameter != NULL)
        return;

      old = foundry_settings_get_action_state (group, action_name);
      parameter = g_variant_new_boolean (!g_variant_get_boolean (old));
      g_variant_unref (old);
    }

  g_action_group_change_action_state (group, action_name, parameter);
}

static const GVariantType *
foundry_settings_get_action_parameter_type (GActionGroup *group,
                                            const char   *action_name)
{
  FoundrySettings *self = FOUNDRY_SETTINGS (group);
  g_autoptr(GSettingsSchemaKey) key = foundry_layered_settings_get_key (self->layered_settings, action_name);
  const GVariantType *type = g_settings_schema_key_get_value_type (key);

  if (g_variant_type_equal (type, G_VARIANT_TYPE_BOOLEAN))
    return NULL;

  return _g_variant_type_intern (type);
}

static void
action_group_iface_init (GActionGroupInterface *iface)
{
  iface->has_action = foundry_settings_has_action;
  iface->list_actions = foundry_settings_list_actions;
  iface->get_action_parameter_type = foundry_settings_get_action_parameter_type;
  iface->get_action_enabled = foundry_settings_get_action_enabled;
  iface->get_action_state = foundry_settings_get_action_state;
  iface->get_action_state_hint = foundry_settings_get_action_state_hint;
  iface->get_action_state_type = foundry_settings_get_action_state_type;
  iface->change_action_state = foundry_settings_change_action_state;
  iface->activate_action = foundry_settings_activate_action;
}

/**
 * foundry_settings_dup_layer:
 * @self: a #FoundrySettings
 * @layer: the desired layer
 *
 * Gets the underlying #GSettings used for the respective layer.
 *
 * Returns: (transfer full): a #GSettings instance
 */
GSettings *
foundry_settings_dup_layer (FoundrySettings      *self,
                            FoundrySettingsLayer  layer)
{
  g_return_val_if_fail (FOUNDRY_IS_SETTINGS (self), NULL);
  g_return_val_if_fail (layer <= FOUNDRY_SETTINGS_LAYER_USER, NULL);

  switch (layer)
    {
    case FOUNDRY_SETTINGS_LAYER_USER:
      return g_object_ref (self->user_settings);

    case FOUNDRY_SETTINGS_LAYER_PROJECT:
      return g_object_ref (self->project_settings);

    case FOUNDRY_SETTINGS_LAYER_APPLICATION:
    default:
      return g_object_ref (self->app_settings);
    }

}
