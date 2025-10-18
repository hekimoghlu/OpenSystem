/* foundry-plugin-manager.c
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

#include "config.h"

#include "foundry-debug.h"
#include "foundry-plugin-manager.h"

#define DISABLED_PLUGINS "disabled-plugins"

struct _FoundryPluginManager
{
  GObject parent_instance;
  GSettings *settings;
  gchar **disabled;
};

G_DEFINE_FINAL_TYPE (FoundryPluginManager, foundry_plugin_manager, G_TYPE_OBJECT)

enum {
  CHANGED,
  N_SIGNALS
};

static guint signals[N_SIGNALS];

static void
_foundry_plugin_manager_load (FoundryPluginManager *self)
{
  PeasEngine *engine;
  guint n_items;

  g_return_if_fail (FOUNDRY_IS_PLUGIN_MANAGER (self));

  engine = peas_engine_get_default ();
  n_items = g_list_model_get_n_items (G_LIST_MODEL (engine));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(PeasPluginInfo) plugin_info = g_list_model_get_item (G_LIST_MODEL (engine), i);
      const char *module_name = peas_plugin_info_get_module_name (plugin_info);

      if (!foundry_plugin_manager_get_disabled (self, plugin_info))
        {
          g_debug ("Loading plugin `%s`", module_name);
          peas_engine_load_plugin (engine, plugin_info);
        }
    }
}

static void
notify_disabled_plugins_cb (FoundryPluginManager *self,
                            const char           *key,
                            GSettings            *settings)
{
  PeasEngine *engine;
  g_auto(GStrv) before = NULL;
  g_auto(GStrv) after = NULL;

  g_assert (FOUNDRY_IS_PLUGIN_MANAGER (self));
  g_assert (G_IS_SETTINGS (settings));

  engine = peas_engine_get_default ();
  before = g_steal_pointer (&self->disabled);
  after = g_settings_get_strv (self->settings, DISABLED_PLUGINS);

  self->disabled = g_strdupv (after);

  for (guint i = 0; before[i]; i++)
    {
      if (!g_strv_contains ((const char * const *)after, before[i]))
        {
          PeasPluginInfo *plugin_info = peas_engine_get_plugin_info (engine, before[i]);

          if (plugin_info != NULL && !peas_plugin_info_is_loaded (plugin_info))
            {
              const char *module_name = peas_plugin_info_get_module_name (plugin_info);

              g_debug ("Loading plugin `%s`", module_name);
              peas_engine_load_plugin (engine, plugin_info);
              g_signal_emit (self, signals[CHANGED], g_quark_from_string (module_name));
            }
        }
    }

  for (guint i = 0; after[i]; i++)
    {
      PeasPluginInfo *plugin_info = peas_engine_get_plugin_info (engine, after[i]);

      if (plugin_info != NULL && peas_plugin_info_is_loaded (plugin_info))
        {
          const char *module_name = peas_plugin_info_get_module_name (plugin_info);

          g_debug ("Unloading plugin `%s`", module_name);
          peas_engine_unload_plugin (engine, plugin_info);
          g_signal_emit (self, signals[CHANGED], g_quark_from_string (module_name));
        }
    }
}

static void
foundry_plugin_manager_finalize (GObject *object)
{
  FoundryPluginManager *self = (FoundryPluginManager *)object;

  g_clear_pointer (&self->disabled, g_strfreev);
  g_clear_object (&self->settings);

  G_OBJECT_CLASS (foundry_plugin_manager_parent_class)->finalize (object);
}

static void
foundry_plugin_manager_class_init (FoundryPluginManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_plugin_manager_finalize;

  signals[CHANGED] =
    g_signal_new ("changed",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST | G_SIGNAL_DETAILED,
                  0,
                  NULL, NULL,
                  NULL,
                  G_TYPE_NONE, 0);
}

static void
foundry_plugin_manager_init (FoundryPluginManager *self)
{
  self->settings = g_settings_new ("app.devsuite.foundry");
  self->disabled = g_settings_get_strv (self->settings, DISABLED_PLUGINS);

  g_signal_connect_object (self->settings,
                           "changed::" DISABLED_PLUGINS,
                           G_CALLBACK (notify_disabled_plugins_cb),
                           self,
                           G_CONNECT_SWAPPED);
}

/**
 * foundry_plugin_manager_get_default:
 *
 * Returns: (transfer none):
 */
FoundryPluginManager *
foundry_plugin_manager_get_default (void)
{
  static FoundryPluginManager *instance;

  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), NULL);

  if (instance == NULL)
    {
      instance = g_object_new (FOUNDRY_TYPE_PLUGIN_MANAGER, NULL);
      g_object_add_weak_pointer (G_OBJECT (instance), (gpointer *)&instance);
      _foundry_plugin_manager_load (instance);
    }

  return instance;
}

gboolean
foundry_plugin_manager_get_disabled (FoundryPluginManager *self,
                                     PeasPluginInfo       *plugin_info)
{
  g_return_val_if_fail (FOUNDRY_IS_PLUGIN_MANAGER (self), FALSE);
  g_return_val_if_fail (PEAS_IS_PLUGIN_INFO (plugin_info), FALSE);

  if (self->disabled == NULL)
    return FALSE;

  return g_strv_contains ((const char * const *)self->disabled,
                          peas_plugin_info_get_module_name (plugin_info));
}

void
foundry_plugin_manager_set_disabled (FoundryPluginManager *self,
                                     PeasPluginInfo       *plugin_info,
                                     gboolean              disabled)
{
  g_autoptr(GStrvBuilder) builder = NULL;
  g_auto(GStrv) strv = NULL;
  const char *name;

  g_return_if_fail (FOUNDRY_IS_PLUGIN_MANAGER (self));
  g_return_if_fail (PEAS_IS_PLUGIN_INFO (plugin_info));

  if (disabled == foundry_plugin_manager_get_disabled (self, plugin_info))
    return;

  name = peas_plugin_info_get_module_name (plugin_info);
  builder = g_strv_builder_new ();

  g_debug ("%s plugin `%s`",
           disabled ? "Disabling" : "Enabling",
           name);

  if (self->disabled)
    {
      for (guint i = 0; self->disabled[i]; i++)
        {
          if (!g_str_equal (name, self->disabled[i]))
            g_strv_builder_add (builder, self->disabled[i]);
        }
    }

  if (disabled)
    g_strv_builder_add (builder, name);

  strv = g_strv_builder_end (builder);

  g_settings_set_strv (self->settings,
                       DISABLED_PLUGINS,
                       (const char * const *)strv);

  g_signal_emit (self, signals[CHANGED], g_quark_from_string (name));
}
