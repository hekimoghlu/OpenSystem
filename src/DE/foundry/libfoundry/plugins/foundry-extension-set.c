/* foundry-extension-set.c
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

#include <stdlib.h>

#include <gobject/gvaluecollector.h>

#include "foundry-debug.h"
#include "foundry-extension-set.h"
#include "foundry-extension-util-private.h"
#include "foundry-marshal.h"

struct _FoundryExtensionSet
{
  FoundryContextual  parent_instance;

  PeasEngine        *engine;
  char              *key;
  char              *value;
  GHashTable        *extensions;
  GPtrArray         *extensions_array;
  GPtrArray         *property_names;
  GArray            *property_values;

  GType              interface_type;

  guint              reload_handler;
};

static GType
foundry_extension_set_get_item_type (GListModel *model)
{
  return FOUNDRY_EXTENSION_SET (model)->interface_type;
}

static guint
foundry_extension_set_get_n_items (GListModel *model)
{
  return FOUNDRY_EXTENSION_SET (model)->extensions_array->len;
}

static gpointer
foundry_extension_set_get_item (GListModel *model,
                                guint       position)
{
  FoundryExtensionSet *self = FOUNDRY_EXTENSION_SET (model);

  if (position < self->extensions_array->len)
    return g_object_ref (g_ptr_array_index (self->extensions_array, position));

  return NULL;
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_extension_set_get_item_type;
  iface->get_n_items = foundry_extension_set_get_n_items;
  iface->get_item = foundry_extension_set_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryExtensionSet, foundry_extension_set, FOUNDRY_TYPE_CONTEXTUAL,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

enum {
  EXTENSIONS_LOADED,
  EXTENSION_ADDED,
  EXTENSION_REMOVED,
  N_SIGNALS
};

enum {
  PROP_0,
  PROP_ENGINE,
  PROP_INTERFACE_TYPE,
  PROP_KEY,
  PROP_VALUE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];
static guint signals[N_SIGNALS];

static void foundry_extension_set_queue_reload (FoundryExtensionSet *);

static void
add_extension (FoundryExtensionSet *self,
               PeasPluginInfo      *plugin_info,
               GObject             *exten)
{
  guint position;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION_SET (self));
  g_assert (plugin_info != NULL);
  g_assert (exten != NULL);
  g_assert (g_type_is_a (G_OBJECT_TYPE (exten), self->interface_type));

  g_hash_table_insert (self->extensions, plugin_info, exten);

  /* Ensure that we take the reference in case it's a floating ref */
  if (G_IS_INITIALLY_UNOWNED (exten) && g_object_is_floating (exten))
    g_object_ref_sink (exten);

  position = self->extensions_array->len;
  g_ptr_array_add (self->extensions_array, exten);
  g_list_model_items_changed (G_LIST_MODEL (self), position, 0, 1);

  g_signal_emit (self, signals [EXTENSION_ADDED], 0, plugin_info, exten);
}

static void
remove_extension (FoundryExtensionSet *self,
                  PeasPluginInfo      *plugin_info,
                  GObject             *exten)
{
  g_autoptr(GObject) hold = NULL;
  guint position;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION_SET (self));
  g_assert (plugin_info != NULL);
  g_assert (exten != NULL);
  g_assert (self->interface_type == G_TYPE_INVALID ||
            g_type_is_a (G_OBJECT_TYPE (exten), self->interface_type));

  hold = g_object_ref (exten);

  g_debug ("Unloading extension %s",
           G_OBJECT_TYPE_NAME (hold));

  g_hash_table_remove (self->extensions, plugin_info);

  if (g_ptr_array_find (self->extensions_array, exten, &position))
    {
      g_ptr_array_remove_index (self->extensions_array, position);
      g_list_model_items_changed (G_LIST_MODEL (self), position, 1, 0);
    }
  else
    {
      g_assert_not_reached ();
    }

  g_signal_emit (self, signals [EXTENSION_REMOVED], 0, plugin_info, hold);
}

static void
foundry_extension_set_reload (FoundryExtensionSet *self)
{
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION_SET (self));
  g_assert (self->interface_type != G_TYPE_INVALID);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->engine));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(PeasPluginInfo) plugin_info = g_list_model_get_item (G_LIST_MODEL (self->engine), i);
      int priority;

      if (!peas_plugin_info_is_loaded (plugin_info))
        continue;

      if (!peas_engine_provides_extension (self->engine, plugin_info, self->interface_type))
        continue;

      if (foundry_extension_util_can_use_plugin (self->engine,
                                                 plugin_info,
                                                 self->interface_type,
                                                 self->key,
                                                 self->value,
                                                 &priority))
        {
          if (!g_hash_table_contains (self->extensions, plugin_info))
            {
              g_autoptr(FoundryContext) context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
              GObject *exten;

              exten = peas_engine_create_extension_with_properties (self->engine,
                                                                    plugin_info,
                                                                    self->interface_type,
                                                                    self->property_names->len,
                                                                    (const char **)self->property_names->pdata,
                                                                    (const GValue *)(gpointer)self->property_values->data);

              if (exten != NULL)
                add_extension (self, plugin_info, exten);
            }
        }
      else
        {
          GObject *exten;

          if ((exten = g_hash_table_lookup (self->extensions, plugin_info)))
            remove_extension (self, plugin_info, exten);
        }
    }

  g_signal_emit (self, signals [EXTENSIONS_LOADED], 0);
}

static gboolean
foundry_extension_set_do_reload (gpointer data)
{
  FoundryExtensionSet *self = data;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION_SET (self));

  self->reload_handler = 0;

  if (self->interface_type != G_TYPE_INVALID)
    foundry_extension_set_reload (self);

  return G_SOURCE_REMOVE;
}

static void
foundry_extension_set_queue_reload (FoundryExtensionSet *self)
{
  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION_SET (self));

  g_clear_handle_id (&self->reload_handler, g_source_remove);

  self->reload_handler = g_idle_add_full (G_PRIORITY_HIGH,
                                          foundry_extension_set_do_reload,
                                          self,
                                          NULL);
}

static void
foundry_extension_set_load_plugin (FoundryExtensionSet *self,
                                   PeasPluginInfo             *plugin_info,
                                   PeasEngine                 *engine)
{
  g_assert (FOUNDRY_IS_EXTENSION_SET (self));
  g_assert (plugin_info != NULL);
  g_assert (PEAS_IS_ENGINE (engine));

  foundry_extension_set_queue_reload (self);
}

static void
foundry_extension_set_unload_plugin (FoundryExtensionSet *self,
                                     PeasPluginInfo      *plugin_info,
                                     PeasEngine          *engine)
{
  GObject *exten;

  g_assert (FOUNDRY_IS_EXTENSION_SET (self));
  g_assert (plugin_info != NULL);
  g_assert (PEAS_IS_ENGINE (engine));

  if ((exten = g_hash_table_lookup (self->extensions, plugin_info)))
    {
      remove_extension (self, plugin_info, exten);
      g_hash_table_remove (self->extensions, plugin_info);
    }
}

static void
foundry_extension_set_set_engine (FoundryExtensionSet *self,
                                  PeasEngine          *engine)
{
  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION_SET (self));
  g_assert (!engine || PEAS_IS_ENGINE (engine));

  if (engine == NULL)
    engine = peas_engine_get_default ();

  if (g_set_object (&self->engine, engine))
    {
      g_signal_connect_object (self->engine, "load-plugin",
                               G_CALLBACK (foundry_extension_set_load_plugin),
                               self,
                               G_CONNECT_AFTER | G_CONNECT_SWAPPED);
      g_signal_connect_object (self->engine, "unload-plugin",
                               G_CALLBACK (foundry_extension_set_unload_plugin),
                               self,
                               G_CONNECT_SWAPPED);
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_ENGINE]);
      foundry_extension_set_queue_reload (self);
    }
}

static void
foundry_extension_set_set_interface_type (FoundryExtensionSet *self,
                                          GType                interface_type)
{
  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION_SET (self));
  g_assert (G_TYPE_IS_INTERFACE (interface_type) || G_TYPE_IS_OBJECT (interface_type));

  if (interface_type != self->interface_type)
    {
      self->interface_type = interface_type;
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_INTERFACE_TYPE]);
      foundry_extension_set_queue_reload (self);
    }
}

static void
foundry_extension_set_dispose (GObject *object)
{
  FoundryExtensionSet *self = (FoundryExtensionSet *)object;
  g_autoptr(GHashTable) extensions = NULL;
  GHashTableIter iter;
  gpointer key;
  gpointer value;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_EXTENSION_SET (self));

  g_clear_pointer (&self->property_names, g_ptr_array_unref);
  g_clear_pointer (&self->property_values, g_array_unref);

  self->interface_type = G_TYPE_INVALID;
  g_clear_handle_id (&self->reload_handler, g_source_remove);

  /*
   * Steal the extensions so we can be re-entrant safe and not break
   * any assumptions about extensions being a real pointer.
   */
  extensions = g_steal_pointer (&self->extensions);
  self->extensions = g_hash_table_new_full (NULL, NULL, NULL, g_object_unref);

  g_hash_table_iter_init (&iter, extensions);

  while (g_hash_table_iter_next (&iter, &key, &value))
    {
      PeasPluginInfo *plugin_info = key;
      GObject *exten = value;

      remove_extension (self, plugin_info, exten);
      g_hash_table_iter_remove (&iter);
    }

  G_OBJECT_CLASS (foundry_extension_set_parent_class)->dispose (object);
}

static void
foundry_extension_set_finalize (GObject *object)
{
  FoundryExtensionSet *self = (FoundryExtensionSet *)object;

  g_clear_object (&self->engine);
  g_clear_pointer (&self->key, g_free);
  g_clear_pointer (&self->value, g_free);
  g_clear_pointer (&self->extensions_array, g_ptr_array_unref);
  g_clear_pointer (&self->extensions, g_hash_table_unref);

  G_OBJECT_CLASS (foundry_extension_set_parent_class)->finalize (object);
}

static void
foundry_extension_set_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundryExtensionSet *self = FOUNDRY_EXTENSION_SET (object);

  switch (prop_id)
    {
    case PROP_ENGINE:
      g_value_set_object (value, foundry_extension_set_get_engine (self));
      break;

    case PROP_INTERFACE_TYPE:
      g_value_set_gtype (value, foundry_extension_set_get_interface_type (self));
      break;

    case PROP_KEY:
      g_value_set_string (value, foundry_extension_set_get_key (self));
      break;

    case PROP_VALUE:
      g_value_set_string (value, foundry_extension_set_get_value (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_extension_set_set_property (GObject      *object,
                                    guint         prop_id,
                                    const GValue *value,
                                    GParamSpec   *pspec)
{
  FoundryExtensionSet *self = FOUNDRY_EXTENSION_SET (object);

  switch (prop_id)
    {
    case PROP_ENGINE:
      foundry_extension_set_set_engine (self, g_value_get_object (value));
      break;

    case PROP_INTERFACE_TYPE:
      foundry_extension_set_set_interface_type (self, g_value_get_gtype (value));
      break;

    case PROP_KEY:
      foundry_extension_set_set_key (self, g_value_get_string (value));
      break;

    case PROP_VALUE:
      foundry_extension_set_set_value (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_extension_set_class_init (FoundryExtensionSetClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_extension_set_dispose;
  object_class->finalize = foundry_extension_set_finalize;
  object_class->get_property = foundry_extension_set_get_property;
  object_class->set_property = foundry_extension_set_set_property;

  properties[PROP_ENGINE] =
    g_param_spec_object ("engine", NULL, NULL,
                         PEAS_TYPE_ENGINE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_INTERFACE_TYPE] =
    g_param_spec_gtype ("interface-type", NULL, NULL,
                        G_TYPE_OBJECT,
                        (G_PARAM_READWRITE |
                         G_PARAM_CONSTRUCT_ONLY |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_KEY] =
    g_param_spec_string ("key", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_VALUE] =
    g_param_spec_string ("value", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  signals[EXTENSION_ADDED] =
    g_signal_new ("extension-added",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  foundry_marshal_VOID__OBJECT_OBJECT,
                  G_TYPE_NONE,
                  2,
                  PEAS_TYPE_PLUGIN_INFO | G_SIGNAL_TYPE_STATIC_SCOPE,
                  G_TYPE_OBJECT | G_SIGNAL_TYPE_STATIC_SCOPE);
  g_signal_set_va_marshaller (signals [EXTENSION_ADDED],
                              G_TYPE_FROM_CLASS (klass),
                              foundry_marshal_VOID__OBJECT_OBJECTv);

  signals[EXTENSION_REMOVED] =
    g_signal_new ("extension-removed",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  foundry_marshal_VOID__OBJECT_OBJECT,
                  G_TYPE_NONE,
                  2,
                  PEAS_TYPE_PLUGIN_INFO | G_SIGNAL_TYPE_STATIC_SCOPE,
                  G_TYPE_OBJECT | G_SIGNAL_TYPE_STATIC_SCOPE);
  g_signal_set_va_marshaller (signals [EXTENSION_REMOVED],
                              G_TYPE_FROM_CLASS (klass),
                              foundry_marshal_VOID__OBJECT_OBJECTv);

  signals[EXTENSIONS_LOADED] =
    g_signal_new ("extensions-loaded",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  foundry_marshal_VOID__VOID,
                  G_TYPE_NONE, 0);
  g_signal_set_va_marshaller (signals [EXTENSIONS_LOADED],
                              G_TYPE_FROM_CLASS (klass),
                              foundry_marshal_VOID__VOIDv);
}

static void
foundry_extension_set_init (FoundryExtensionSet *self)
{
  self->extensions = g_hash_table_new_full (NULL, NULL, NULL, g_object_unref);
  self->extensions_array = g_ptr_array_new ();
}

/**
 * foundry_extension_set_get_engine:
 *
 * Gets the #FoundryExtensionSet:engine property.
 *
 * Returns: (transfer none): a #PeasEngine.
 */
PeasEngine *
foundry_extension_set_get_engine (FoundryExtensionSet *self)
{
  g_return_val_if_fail (FOUNDRY_IS_EXTENSION_SET (self), NULL);

  return self->engine;
}

GType
foundry_extension_set_get_interface_type (FoundryExtensionSet *self)
{
  g_return_val_if_fail (FOUNDRY_IS_EXTENSION_SET (self), G_TYPE_INVALID);

  return self->interface_type;
}

const char *
foundry_extension_set_get_key (FoundryExtensionSet *self)
{
  g_return_val_if_fail (FOUNDRY_IS_EXTENSION_SET (self), NULL);

  return self->key;
}

void
foundry_extension_set_set_key (FoundryExtensionSet *self,
                               const char          *key)
{
  g_return_if_fail (FOUNDRY_IS_MAIN_THREAD ());
  g_return_if_fail (FOUNDRY_IS_EXTENSION_SET (self));

  if (g_set_str (&self->key, key))
    {
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_KEY]);
      foundry_extension_set_queue_reload (self);
    }
}

const char *
foundry_extension_set_get_value (FoundryExtensionSet *self)
{
  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), NULL);
  g_return_val_if_fail (FOUNDRY_IS_EXTENSION_SET (self), NULL);

  return self->value;
}

void
foundry_extension_set_set_value (FoundryExtensionSet *self,
                                 const char          *value)
{
  g_return_if_fail (FOUNDRY_IS_MAIN_THREAD ());
  g_return_if_fail (FOUNDRY_IS_EXTENSION_SET (self));

  g_debug ("Setting extension adapter %s value to \"%s\"",
           g_type_name (self->interface_type),
           value ?: "");

  if (g_set_str (&self->value, value))
    {
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_VALUE]);
      foundry_extension_set_queue_reload (self);
    }
}

/**
 * foundry_extension_set_foreach:
 * @self: an #FoundryExtensionSet
 * @foreach_func: (scope call): A callback
 * @user_data: user data for @foreach_func
 *
 * Calls @foreach_func for every extension loaded by the extension set.
 */
void
foundry_extension_set_foreach (FoundryExtensionSet            *self,
                               FoundryExtensionSetForeachFunc  foreach_func,
                               gpointer                        user_data)
{
  guint n_items;

  g_return_if_fail (FOUNDRY_IS_EXTENSION_SET (self));
  g_return_if_fail (foreach_func != NULL);

  /*
   * Use the ordered list of plugins as it is sorted including any
   * dependencies of plugins.
   */

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->engine));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(PeasPluginInfo) plugin_info = g_list_model_get_item (G_LIST_MODEL (self->engine), i);
      GObject *exten = g_hash_table_lookup (self->extensions, plugin_info);

      if (exten != NULL)
        foreach_func (self, plugin_info, exten, user_data);
    }
}

typedef struct
{
  PeasPluginInfo *plugin_info;
  GObject        *exten;
  int             priority;
} SortedInfo;

static gint
sort_by_priority (gconstpointer a,
                  gconstpointer b)
{
  const SortedInfo *sa = a;
  const SortedInfo *sb = b;

  /* Lower values are higher priority */

  if (sa->priority < sb->priority)
    return -1;
  else if (sa->priority > sb->priority)
    return 1;
  else
    return 0;
}

/**
 * foundry_extension_set_foreach_by_priority:
 * @self: an #FoundryExtensionSet
 * @foreach_func: (scope call): A callback
 * @user_data: user data for @foreach_func
 *
 * Calls @foreach_func for every extension loaded by the extension set.
 */
void
foundry_extension_set_foreach_by_priority (FoundryExtensionSet            *self,
                                           FoundryExtensionSetForeachFunc  foreach_func,
                                           gpointer                        user_data)
{
  g_autoptr(GArray) sorted = NULL;
  g_autofree char *prio_key = NULL;
  GHashTableIter iter;
  gpointer key;
  gpointer value;

  g_return_if_fail (FOUNDRY_IS_MAIN_THREAD ());
  g_return_if_fail (FOUNDRY_IS_EXTENSION_SET (self));
  g_return_if_fail (foreach_func != NULL);

  if (self->key == NULL)
    {
      foundry_extension_set_foreach (self, foreach_func, user_data);
      return;
    }

  prio_key = g_strdup_printf ("%s-Priority", self->key);
  sorted = g_array_new (FALSE, FALSE, sizeof (SortedInfo));

  g_hash_table_iter_init (&iter, self->extensions);

  while (g_hash_table_iter_next (&iter, &key, &value))
    {
      PeasPluginInfo *plugin_info = key;
      GObject *exten = value;
      const char *priostr = peas_plugin_info_get_external_data (plugin_info, prio_key);
      gint prio = priostr ? atoi (priostr) : 0;
      SortedInfo info = { plugin_info, exten, prio };

      g_array_append_val (sorted, info);
    }

  g_array_sort (sorted, sort_by_priority);

  for (guint i = 0; i < sorted->len; i++)
    {
      const SortedInfo *info = &g_array_index (sorted, SortedInfo, i);

      foreach_func (self, info->plugin_info, info->exten, user_data);
    }
}

guint
foundry_extension_set_get_n_extensions (FoundryExtensionSet *self)
{
  g_return_val_if_fail (FOUNDRY_IS_EXTENSION_SET (self), 0);

  if (self->extensions != NULL)
    return g_hash_table_size (self->extensions);

  return 0;
}

FoundryExtensionSet *
foundry_extension_set_new (FoundryContext *context,
                           PeasEngine     *engine,
                           GType           interface_type,
                           const char     *key,
                           const char     *value,
                           ...)
{
  g_autoptr(GTypeClass) type_class = NULL;
  g_autoptr(GPtrArray) property_names = NULL;
  g_autoptr(GArray) property_values = NULL;
  FoundryExtensionSet *ret;
  const char *name;
  va_list args;

  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), NULL);
  g_return_val_if_fail (!context || FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (!engine || PEAS_IS_ENGINE (engine), NULL);
  g_return_val_if_fail (G_TYPE_IS_INTERFACE (interface_type) ||
                        G_TYPE_IS_OBJECT (interface_type), NULL);

  type_class = g_type_class_ref (interface_type);

  property_names = g_ptr_array_new_null_terminated (0, NULL, TRUE);
  property_values = g_array_new (FALSE, FALSE, sizeof (GValue));
  g_array_set_clear_func (property_values, (GDestroyNotify)g_value_unset);

  if (context != NULL &&
      g_type_is_a (interface_type, FOUNDRY_TYPE_CONTEXTUAL))
    {
      GValue gvalue = G_VALUE_INIT;

      g_ptr_array_add (property_names, (gpointer)"context");

      g_value_init (&gvalue, FOUNDRY_TYPE_CONTEXT);
      g_value_set_object (&gvalue, context);

      g_array_append_val (property_values, gvalue);
    }

  va_start (args, value);
  while ((name = va_arg (args, const char *)))
    {
      GParamSpec *pspec = g_object_class_find_property (G_OBJECT_CLASS (type_class), name);
      g_autofree char *errmsg = NULL;
      GValue gvalue = G_VALUE_INIT;

      if (pspec == NULL)
        g_error ("`%s` has no such property `%s`",
                 g_type_name (interface_type), name);

      g_ptr_array_add (property_names, (gpointer)g_intern_string (name));

      g_value_init (&gvalue, pspec->value_type);
      G_VALUE_COLLECT (&gvalue, args, 0, &errmsg);

      if (errmsg != NULL)
        g_error ("`%s` failed to collect value for `%s`: %s",
                 g_type_name (interface_type), name, errmsg);

      g_array_append_val (property_values, gvalue);
    }
  va_end (args);

  ret = g_object_new (FOUNDRY_TYPE_EXTENSION_SET,
                      "context", context,
                      "engine", engine,
                      "interface-type", interface_type,
                      "key", key,
                      "value", value,
                      NULL);

  ret->property_names = g_steal_pointer (&property_names);
  ret->property_values = g_steal_pointer (&property_values);

  /* If we have a reload queued, just process it immediately so that
   * there is some determinism in plugin loading.
   */
  if (ret->reload_handler != 0)
    {
      g_clear_handle_id (&ret->reload_handler, g_source_remove);
      foundry_extension_set_do_reload (ret);
    }

  return ret;
}

/**
 * foundry_extension_set_get_extension:
 * @self: a #FoundryExtensionSet
 * @plugin_info: a #PeasPluginInfo
 *
 * Locates the extension owned by @plugin_info if such extension exists.
 *
 * Returns: (transfer none) (nullable): a #GObject or %NULL
 */
GObject *
foundry_extension_set_get_extension (FoundryExtensionSet *self,
                                     PeasPluginInfo      *plugin_info)
{
  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), NULL);
  g_return_val_if_fail (FOUNDRY_IS_EXTENSION_SET (self), NULL);
  g_return_val_if_fail (plugin_info != NULL, NULL);

  return g_hash_table_lookup (self->extensions, plugin_info);
}
