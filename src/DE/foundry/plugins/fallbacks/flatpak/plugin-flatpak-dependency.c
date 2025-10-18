/* plugin-flatpak-dependency.c
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

#include <glib/gstdio.h>

#include "plugin-flatpak-config.h"
#include "plugin-flatpak-dependency.h"
#include "plugin-flatpak-util.h"

struct _PluginFlatpakDependency
{
  FoundryDependency     parent_instance;
  FoundryFlatpakModule *module;
};

G_DEFINE_FINAL_TYPE (PluginFlatpakDependency, plugin_flatpak_dependency, FOUNDRY_TYPE_DEPENDENCY)

enum {
  PROP_0,
  PROP_MODULE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static FoundryFlatpakSource *
get_first_source (FoundryFlatpakModule *module)
{
  g_autoptr(FoundryFlatpakSources) sources = NULL;
  g_autoptr(FoundryFlatpakModules) submodules = NULL;

  if (module == NULL)
    return NULL;

  if ((sources = foundry_flatpak_module_dup_sources (module)))
    {
      if (g_list_model_get_n_items (G_LIST_MODEL (sources)) > 0)
        return g_list_model_get_item (G_LIST_MODEL (sources), 0);
    }

  if ((submodules = foundry_flatpak_module_dup_modules (module)))
    {
      guint n_items = g_list_model_get_n_items (G_LIST_MODEL (submodules));

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryFlatpakModule) submodule = g_list_model_get_item (G_LIST_MODEL (submodules), i);
          g_autoptr(FoundryFlatpakSource) source = get_first_source (submodule);

          if (source != NULL)
            return g_steal_pointer (&source);
        }
    }

  return NULL;
}

static char *
plugin_flatpak_dependency_dup_name (FoundryDependency *dependency)
{
  return foundry_flatpak_module_dup_name (PLUGIN_FLATPAK_DEPENDENCY (dependency)->module);
}

static char *
plugin_flatpak_dependency_dup_kind (FoundryDependency *dependency)
{
  PluginFlatpakDependency *self = PLUGIN_FLATPAK_DEPENDENCY (dependency);
  g_autoptr(FoundryFlatpakSource) source = get_first_source (self->module);

  if (source != NULL)
    return g_strdup (FOUNDRY_FLATPAK_SOURCE_GET_CLASS (source)->type);

  return g_strdup ("flatpak");
}

static char *
plugin_flatpak_dependency_dup_location (FoundryDependency *dependency)
{
  PluginFlatpakDependency *self = PLUGIN_FLATPAK_DEPENDENCY (dependency);
  g_autoptr(FoundryFlatpakSource) source = get_first_source (self->module);

  if (source != NULL)
    {
      const char *prop_name = NULL;
      const char *backup_prop_name = NULL;
      char *value = NULL;

      if (0) {}
      else if (FOUNDRY_IS_FLATPAK_SOURCE_ARCHIVE (source))
        prop_name = "url", backup_prop_name = "path";
      else if (FOUNDRY_IS_FLATPAK_SOURCE_BZR (source))
        prop_name = "url";
      else if (FOUNDRY_IS_FLATPAK_SOURCE_DIR (source))
        prop_name = "path";
      else if (FOUNDRY_IS_FLATPAK_SOURCE_EXTRA_DATA (source))
        prop_name = "url", backup_prop_name = "filename";
      else if (FOUNDRY_IS_FLATPAK_SOURCE_FILE (source))
        prop_name = "url", backup_prop_name = "path";
      else if (FOUNDRY_IS_FLATPAK_SOURCE_GIT (source))
        prop_name = "url", backup_prop_name = "path";
      else if (FOUNDRY_IS_FLATPAK_SOURCE_INLINE (source))
        prop_name = NULL;
      else if (FOUNDRY_IS_FLATPAK_SOURCE_PATCH (source))
        prop_name = "path";
      else if (FOUNDRY_IS_FLATPAK_SOURCE_SCRIPT (source))
        prop_name = NULL;
      else if (FOUNDRY_IS_FLATPAK_SOURCE_SHELL (source))
        prop_name = NULL;
      else if (FOUNDRY_IS_FLATPAK_SOURCE_SVN (source))
        prop_name = "url";

      if (prop_name != NULL)
        {
          g_object_get (source, prop_name, &value, NULL);

          if (!value && backup_prop_name != NULL)
            g_object_get (source, backup_prop_name, &value, NULL);
        }

      return g_steal_pointer (&value);
    }

  return NULL;
}

static void
plugin_flatpak_dependency_finalize (GObject *object)
{
  PluginFlatpakDependency *self = (PluginFlatpakDependency *)object;

  g_clear_object (&self->module);

  G_OBJECT_CLASS (plugin_flatpak_dependency_parent_class)->finalize (object);
}

static void
plugin_flatpak_dependency_get_property (GObject    *object,
                                        guint       prop_id,
                                        GValue     *value,
                                        GParamSpec *pspec)
{
  PluginFlatpakDependency *self = PLUGIN_FLATPAK_DEPENDENCY (object);

  switch (prop_id)
    {
    case PROP_MODULE:
      g_value_set_object (value, self->module);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_dependency_set_property (GObject      *object,
                                        guint         prop_id,
                                        const GValue *value,
                                        GParamSpec   *pspec)
{
  PluginFlatpakDependency *self = PLUGIN_FLATPAK_DEPENDENCY (object);

  switch (prop_id)
    {
    case PROP_MODULE:
      self->module = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_dependency_class_init (PluginFlatpakDependencyClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDependencyClass *dependency_class = FOUNDRY_DEPENDENCY_CLASS (klass);

  object_class->finalize = plugin_flatpak_dependency_finalize;
  object_class->get_property = plugin_flatpak_dependency_get_property;
  object_class->set_property = plugin_flatpak_dependency_set_property;

  dependency_class->dup_name = plugin_flatpak_dependency_dup_name;
  dependency_class->dup_kind = plugin_flatpak_dependency_dup_kind;
  dependency_class->dup_location = plugin_flatpak_dependency_dup_location;

  properties[PROP_MODULE] =
    g_param_spec_object ("module", NULL, NULL,
                         FOUNDRY_TYPE_FLATPAK_MODULE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_flatpak_dependency_init (PluginFlatpakDependency *self)
{
}

PluginFlatpakDependency *
plugin_flatpak_dependency_new (FoundryContext            *context,
                               FoundryDependencyProvider *provider,
                               FoundryFlatpakModule      *module)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MODULE (module), NULL);
  g_return_val_if_fail (FOUNDRY_IS_DEPENDENCY_PROVIDER (provider), NULL);

  return g_object_new (PLUGIN_TYPE_FLATPAK_DEPENDENCY,
                       "context", context,
                       "module", module,
                       "provider", provider,
                       NULL);
}
