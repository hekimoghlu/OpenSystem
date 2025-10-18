/* foundry-extension.c
 *
 * Copyright 2015-2024 Christian Hergert <chergert@redhat.com>
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

#include "foundry-debug.h"
#include "foundry-extension.h"
#include "foundry-extension-util-private.h"

struct _FoundryExtension
{
  FoundryContextual  parent_instance;

  PeasEngine        *engine;
  char              *key;
  char              *value;
  GObject           *extension;

  PeasPluginInfo    *plugin_info;

  GType              interface_type;
  guint              queue_handler;
};

G_DEFINE_FINAL_TYPE (FoundryExtension, foundry_extension, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_ENGINE,
  PROP_EXTENSION,
  PROP_INTERFACE_TYPE,
  PROP_KEY,
  PROP_VALUE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_extension_set_extension (FoundryExtension *self,
                                 PeasPluginInfo   *plugin_info,
                                 GObject          *extension)
{
  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION (self));
  g_assert (!extension || self->interface_type != G_TYPE_INVALID);
  g_assert (!extension || g_type_is_a (G_OBJECT_TYPE (extension), self->interface_type));

  self->plugin_info = plugin_info;

  if (g_set_object (&self->extension, extension))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_EXTENSION]);
}

static void
foundry_extension_reload (FoundryExtension *self)
{
  GObject  *extension = NULL;
  PeasPluginInfo *best_match = NULL;
  guint n_items;
  int best_match_priority = G_MININT;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION (self));
  g_assert (self->interface_type != G_TYPE_INVALID);

  if (!self->engine || !self->key || !self->value)
    {
      foundry_extension_set_extension (self, NULL, NULL);
      return;
    }

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->engine));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(PeasPluginInfo) plugin_info = g_list_model_get_item (G_LIST_MODEL (self->engine), i);
      int priority = 0;

      if (foundry_extension_util_can_use_plugin (self->engine,
                                                 plugin_info,
                                                 self->interface_type,
                                                 self->key,
                                                 self->value,
                                                 &priority))
        {
          if (priority > best_match_priority)
            {
              best_match = plugin_info;
              best_match_priority = priority;
            }
        }
    }

  g_debug ("Best match for %s=%s is %s",
           self->key, self->value,
           best_match ? peas_plugin_info_get_name (best_match) : "no match");

  /*
   * If the desired extension matches our already loaded extension, then
   * ignore the attempt to create a new instance of the extension.
   */
  if ((self->extension != NULL) && (best_match != NULL) && (best_match == self->plugin_info))
    return;

  if (best_match != NULL)
    {
      g_autoptr(FoundryContext) context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

      if (g_type_is_a (self->interface_type, FOUNDRY_TYPE_CONTEXTUAL))
        extension = peas_engine_create_extension (self->engine,
                                                  best_match,
                                                  self->interface_type,
                                                  "context", context,
                                                  NULL);
      else
        extension = peas_engine_create_extension (self->engine,
                                                  best_match,
                                                  self->interface_type,
                                                  NULL);
    }

  foundry_extension_set_extension (self, best_match, extension);

  g_clear_object (&extension);
}

static gboolean
foundry_extension_do_reload (gpointer data)
{
  FoundryExtension *self = data;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION (self));

  self->queue_handler = 0;

  if (self->interface_type != G_TYPE_INVALID)
    foundry_extension_reload (self);

  return G_SOURCE_REMOVE;
}

static void
foundry_extension_queue_reload (FoundryExtension *self)
{
  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION (self));

  g_clear_handle_id (&self->queue_handler, g_source_remove);
  self->queue_handler = g_timeout_add (0, foundry_extension_do_reload, self);
}

static void
foundry_extension__engine_load_plugin (FoundryExtension *self,
                                       PeasPluginInfo          *plugin_info,
                                       PeasEngine              *engine)
{
  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION (self));
  g_assert (plugin_info != NULL);
  g_assert (PEAS_IS_ENGINE (engine));

  if (peas_engine_provides_extension (self->engine, plugin_info, self->interface_type))
    foundry_extension_queue_reload (self);
}

static void
foundry_extension__engine_unload_plugin (FoundryExtension *self,
                                         PeasPluginInfo          *plugin_info,
                                         PeasEngine              *engine)
{
  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION (self));
  g_assert (plugin_info != NULL);
  g_assert (PEAS_IS_ENGINE (engine));

  if (self->extension != NULL)
    {
      if (plugin_info == self->plugin_info)
        {
          g_clear_object (&self->extension);
          foundry_extension_queue_reload (self);
        }
    }
}

static void
foundry_extension_set_engine (FoundryExtension *self,
                              PeasEngine              *engine)
{
  g_return_if_fail (FOUNDRY_IS_MAIN_THREAD ());
  g_return_if_fail (FOUNDRY_IS_EXTENSION (self));
  g_return_if_fail (!engine || PEAS_IS_ENGINE (engine));
  g_return_if_fail (self->engine == NULL);

  if (engine == NULL)
    engine = peas_engine_get_default ();

  self->engine = g_object_ref (engine);

  g_signal_connect_object (self->engine,
                           "load-plugin",
                           G_CALLBACK (foundry_extension__engine_load_plugin),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (self->engine,
                           "unload-plugin",
                           G_CALLBACK (foundry_extension__engine_unload_plugin),
                           self,
                           G_CONNECT_SWAPPED);

  foundry_extension_queue_reload (self);
}

static void
foundry_extension_set_interface_type (FoundryExtension *self,
                                      GType             interface_type)
{
  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION (self));
  g_assert (G_TYPE_IS_INTERFACE (interface_type) || G_TYPE_IS_ABSTRACT (interface_type));

  if (self->interface_type != interface_type)
    {
      self->interface_type = interface_type;
      foundry_extension_queue_reload (self);
    }
}

static void
foundry_extension_dispose (GObject *object)
{
  FoundryExtension *self = (FoundryExtension *)object;

  self->interface_type = G_TYPE_INVALID;

  g_clear_handle_id (&self->queue_handler, g_source_remove);

  G_OBJECT_CLASS (foundry_extension_parent_class)->dispose (object);
}

static void
foundry_extension_finalize (GObject *object)
{
  FoundryExtension *self = (FoundryExtension *)object;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (self->interface_type == G_TYPE_INVALID);
  g_assert (self->queue_handler == 0);

  g_clear_object (&self->extension);
  g_clear_object (&self->engine);
  g_clear_pointer (&self->key, g_free);
  g_clear_pointer (&self->value, g_free);

  G_OBJECT_CLASS (foundry_extension_parent_class)->finalize (object);
}

static void
foundry_extension_get_property (GObject    *object,
                                guint       prop_id,
                                GValue     *value,
                                GParamSpec *pspec)
{
  FoundryExtension *self = FOUNDRY_EXTENSION (object);

  switch (prop_id)
    {
    case PROP_ENGINE:
      g_value_set_object (value, foundry_extension_get_engine (self));
      break;

    case PROP_EXTENSION:
      g_value_set_object (value, foundry_extension_get_extension (self));
      break;

    case PROP_INTERFACE_TYPE:
      g_value_set_gtype (value, foundry_extension_get_interface_type (self));
      break;

    case PROP_KEY:
      g_value_set_string (value, foundry_extension_get_key (self));
      break;

    case PROP_VALUE:
      g_value_set_string (value, foundry_extension_get_value (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_extension_set_property (GObject      *object,
                                guint         prop_id,
                                const GValue *value,
                                GParamSpec   *pspec)
{
  FoundryExtension *self = FOUNDRY_EXTENSION (object);

  switch (prop_id)
    {
    case PROP_ENGINE:
      foundry_extension_set_engine (self, g_value_get_object (value));
      break;

    case PROP_INTERFACE_TYPE:
      foundry_extension_set_interface_type (self, g_value_get_gtype (value));
      break;

    case PROP_KEY:
      foundry_extension_set_key (self, g_value_get_string (value));
      break;

    case PROP_VALUE:
      foundry_extension_set_value (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_extension_class_init (FoundryExtensionClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_extension_dispose;
  object_class->finalize = foundry_extension_finalize;
  object_class->get_property = foundry_extension_get_property;
  object_class->set_property = foundry_extension_set_property;

  properties[PROP_ENGINE] =
    g_param_spec_object ("engine",
                         "Engine",
                         "Engine",
                         PEAS_TYPE_ENGINE,
                         (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS));

  properties[PROP_EXTENSION] =
    g_param_spec_object ("extension",
                         "Extension",
                         "The extension object.",
                         G_TYPE_OBJECT,
                         (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  properties[PROP_INTERFACE_TYPE] =
    g_param_spec_gtype ("interface-type",
                        "Interface Type",
                        "The GType of the extension interface.",
                        G_TYPE_OBJECT,
                        (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS));

  properties[PROP_KEY] =
    g_param_spec_string ("key",
                         "Key",
                         "The external data key to match from plugin info.",
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  properties[PROP_VALUE] =
    g_param_spec_string ("value",
                         "Value",
                         "The external data value to match from plugin info.",
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties(object_class, N_PROPS, properties);
}

static void
foundry_extension_init (FoundryExtension *self)
{
  self->interface_type = G_TYPE_INVALID;
}

const char *
foundry_extension_get_key (FoundryExtension *self)
{
  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), NULL);
  g_return_val_if_fail (FOUNDRY_IS_EXTENSION (self), NULL);

  return self->key;
}

void
foundry_extension_set_key (FoundryExtension *self,
                           const char       *key)
{
  g_return_if_fail (FOUNDRY_IS_MAIN_THREAD ());
  g_return_if_fail (FOUNDRY_IS_EXTENSION (self));

  if (g_set_str (&self->key, key))
    {
      foundry_extension_queue_reload (self);
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_KEY]);
    }
}

const char *
foundry_extension_get_value (FoundryExtension *self)
{
  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), NULL);
  g_return_val_if_fail (FOUNDRY_IS_EXTENSION (self), NULL);

  return self->value;
}

void
foundry_extension_set_value (FoundryExtension *self,
                             const char       *value)
{
  g_return_if_fail (FOUNDRY_IS_MAIN_THREAD ());
  g_return_if_fail (FOUNDRY_IS_EXTENSION (self));

  if (g_set_str (&self->value, value))
    {
      if (self->interface_type != G_TYPE_INVALID)
        foundry_extension_reload (self);
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_VALUE]);
    }
}

GType
foundry_extension_get_interface_type (FoundryExtension *self)
{
  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), G_TYPE_INVALID);
  g_return_val_if_fail (FOUNDRY_IS_EXTENSION (self), G_TYPE_INVALID);

  return self->interface_type;
}

/**
 * foundry_extension_get_engine:
 *
 * Gets the #FoundryExtension:engine property.
 *
 * Returns: (transfer none): a #PeasEngine.
 */
PeasEngine *
foundry_extension_get_engine (FoundryExtension *self)
{
  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), NULL);
  g_return_val_if_fail (FOUNDRY_IS_EXTENSION (self), NULL);

  return self->engine;
}

/**
 * foundry_extension_get_extension:
 *
 * Gets the extension object managed by the adapter.
 *
 * Returns: (transfer none) (type GObject.Object): a #GObject or %NULL.
 */
gpointer
foundry_extension_get_extension (FoundryExtension *self)
{
  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), NULL);
  g_return_val_if_fail (FOUNDRY_IS_EXTENSION (self), NULL);

  /* If we have a reload queued, then immediately perform the reload now
   * that the user is requesting it.
   */
  if (self->queue_handler > 0)
    {
      g_clear_handle_id (&self->queue_handler, g_source_remove);
      foundry_extension_reload (self);
    }

  return self->extension;
}

/**
 * foundry_extension_new:
 * @context: (nullable): An #FoundryContext or %NULL
 * @engine: (allow-none): a #PeasEngine or %NULL
 * @interface_type: The #GType of the interface to be implemented.
 * @key: The key for matching extensions from plugin info external data.
 * @value: (allow-none): The value to use when matching keys.
 *
 * Creates a new #FoundryExtension.
 *
 * The #FoundryExtension object can be used to wrap an extension that might
 * need to change at runtime based on various changing parameters. For example,
 * it can watch the loading and unloading of plugins and reload the
 * #FoundryExtension:extension property.
 *
 * Additionally, it can match a specific plugin based on the @value provided.
 *
 * Returns: (transfer full): A newly created #FoundryExtension.
 */
FoundryExtension *
foundry_extension_new (FoundryContext *context,
                       PeasEngine     *engine,
                       GType           interface_type,
                       const char     *key,
                       const char     *value)
{
  FoundryExtension *self;

  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), NULL);
  g_return_val_if_fail (!context || FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (!engine || PEAS_IS_ENGINE (engine), NULL);
  g_return_val_if_fail (G_TYPE_IS_INTERFACE (interface_type) || G_TYPE_IS_ABSTRACT (interface_type), NULL);
  g_return_val_if_fail (key != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_EXTENSION,
                       "context", context,
                       "engine", engine,
                       "interface-type", interface_type,
                       "key", key,
                       "value", value,
                       NULL);

  return g_steal_pointer (&self);
}
