/* plugin-flatpak-config.c
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

#include "plugin-flatpak-config.h"

#define PRIORITY_DEFAULT     100
#define PRIORITY_MAYBE_DEVEL 200
#define PRIORITY_DEVEL       300

struct _PluginFlatpakConfig
{
  FoundryConfig           parent_instance;
  FoundryFlatpakManifest *manifest;
  GFile                  *file;
};

enum {
  PROP_0,
  PROP_FILE,
  PROP_MANIFEST,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (PluginFlatpakConfig, plugin_flatpak_config, FOUNDRY_TYPE_CONFIG)

static GParamSpec *properties[N_PROPS];

static gboolean
plugin_flatpak_config_can_default (FoundryConfig *config,
                                   guint         *priority)
{
  PluginFlatpakConfig *self = PLUGIN_FLATPAK_CONFIG (config);
  g_autofree char *name = NULL;

  if (!(name = g_file_get_basename (self->file)))
    return FALSE;

  *priority = PRIORITY_DEFAULT;

  if (strstr (name, "Devel") != NULL)
    *priority = PRIORITY_MAYBE_DEVEL;

  if (strstr (name, ".Devel.") != NULL)
    *priority = PRIORITY_DEVEL;

  return TRUE;
}

static char *
plugin_flatpak_config_dup_build_system (FoundryConfig *config)
{
  PluginFlatpakConfig *self = (PluginFlatpakConfig *)config;
  g_autoptr(FoundryFlatpakModule) primary_module = NULL;

  g_assert (PLUGIN_IS_FLATPAK_CONFIG (self));

  if ((primary_module = plugin_flatpak_config_dup_primary_module (self)))
    {
      g_autofree char *buildsystem = NULL;

      buildsystem = foundry_flatpak_module_dup_buildsystem (primary_module);

      if (g_strcmp0 (buildsystem, "simple") == 0)
        return g_strdup ("flatpak-simple");

      return g_steal_pointer (&buildsystem);
    }

  return NULL;
}

static char **
plugin_flatpak_config_dup_config_opts (FoundryConfig *config)
{
  PluginFlatpakConfig *self = (PluginFlatpakConfig *)config;
  g_autoptr(FoundryFlatpakModule) primary_module = NULL;

  g_assert (PLUGIN_IS_FLATPAK_CONFIG (self));

  if ((primary_module = plugin_flatpak_config_dup_primary_module (self)))
    return foundry_flatpak_module_dup_config_opts (primary_module);

  return NULL;
}

static FoundryCommand *
plugin_flatpak_config_dup_default_command (FoundryConfig *config)
{
  PluginFlatpakConfig *self = (PluginFlatpakConfig *)config;
  g_autofree char *argv0 = NULL;
  g_autoptr(FoundryCommand) command = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GStrvBuilder) builder = NULL;
  g_auto(GStrv) argv = NULL;
  g_auto(GStrv) x_run_args = NULL;

  g_assert (PLUGIN_IS_FLATPAK_CONFIG (self));

  argv0 = foundry_flatpak_manifest_dup_command (self->manifest);
  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  command = foundry_command_new (context);

  builder = g_strv_builder_new ();
  g_strv_builder_add (builder, argv0);

  /* x-run-args */
  if ((x_run_args = foundry_flatpak_serializable_dup_x_strv (FOUNDRY_FLATPAK_SERIALIZABLE (self->manifest), "x-run-args")))
    g_strv_builder_addv (builder, (const char **)x_run_args);

  argv = g_strv_builder_end (builder);
  foundry_command_set_argv (command, (const char * const *)argv);

  return g_steal_pointer (&command);
}

static DexFuture *
plugin_flatpak_config_resolve_sdk_fiber (gpointer data)
{
  FoundryPair *pair = data;
  PluginFlatpakConfig *self = PLUGIN_FLATPAK_CONFIG (pair->first);
  FoundryDevice *device = FOUNDRY_DEVICE (pair->second);
  g_autoptr(FoundrySdkManager) sdk_manager = NULL;
  g_autoptr(FoundryDeviceInfo) device_info = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryTriplet) triplet = NULL;
  g_autoptr(FoundrySdk) sdk = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *id = NULL;
  g_autofree char *runtime = NULL;
  g_autofree char *runtime_version = NULL;
  g_autofree char *sdk_str = NULL;
  const char *arch;

  g_assert (PLUGIN_IS_FLATPAK_CONFIG (self));
  g_assert (FOUNDRY_IS_DEVICE (device));

  g_object_get (self->manifest,
                "runtime", &runtime,
                "runtime-version", &runtime_version,
                "sdk", &sdk_str,
                NULL);

  if (runtime == NULL || runtime_version == NULL)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "Manifest is missing information required to resolve SDK");

  if (sdk_str != NULL)
    g_set_str (&runtime, sdk_str);

  if (!(device_info = dex_await_object (foundry_device_load_info (device), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  triplet = foundry_device_info_dup_triplet (device_info);
  arch = foundry_triplet_get_arch (triplet);
  id = g_strdup_printf ("%s/%s/%s", runtime, arch, runtime_version);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  sdk_manager = foundry_context_dup_sdk_manager (context);

  return foundry_sdk_manager_find_by_id (sdk_manager, id);
}

static DexFuture *
plugin_flatpak_config_resolve_sdk (FoundryConfig *config,
                                   FoundryDevice *device)
{
  PluginFlatpakConfig *self = (PluginFlatpakConfig *)config;

  g_assert (PLUGIN_IS_FLATPAK_CONFIG (self));
  g_assert (FOUNDRY_IS_DEVICE (device));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_flatpak_config_resolve_sdk_fiber,
                              foundry_pair_new (self, device),
                              (GDestroyNotify) foundry_pair_free);
}

static void
plugin_flatpak_config_finalize (GObject *object)
{
  PluginFlatpakConfig *self = (PluginFlatpakConfig *)object;

  g_clear_object (&self->file);
  g_clear_object (&self->manifest);

  G_OBJECT_CLASS (plugin_flatpak_config_parent_class)->finalize (object);
}

static void
plugin_flatpak_config_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  PluginFlatpakConfig *self = PLUGIN_FLATPAK_CONFIG (object);

  switch (prop_id)
    {
    case PROP_FILE:
      g_value_take_object (value, plugin_flatpak_config_dup_file (self));
      break;

    case PROP_MANIFEST:
      g_value_take_object (value, plugin_flatpak_config_dup_manifest (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_config_set_property (GObject      *object,
                                    guint         prop_id,
                                    const GValue *value,
                                    GParamSpec   *pspec)
{
  PluginFlatpakConfig *self = PLUGIN_FLATPAK_CONFIG (object);

  switch (prop_id)
    {
    case PROP_FILE:
      self->file = g_value_dup_object (value);
      break;

    case PROP_MANIFEST:
      self->manifest = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_config_class_init (PluginFlatpakConfigClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryConfigClass *config_class = FOUNDRY_CONFIG_CLASS (klass);

  object_class->finalize = plugin_flatpak_config_finalize;
  object_class->get_property = plugin_flatpak_config_get_property;
  object_class->set_property = plugin_flatpak_config_set_property;

  config_class->dup_build_system = plugin_flatpak_config_dup_build_system;
  config_class->dup_config_opts = plugin_flatpak_config_dup_config_opts;
  config_class->dup_default_command = plugin_flatpak_config_dup_default_command;
  config_class->can_default = plugin_flatpak_config_can_default;
  config_class->resolve_sdk = plugin_flatpak_config_resolve_sdk;

  properties[PROP_FILE] =
    g_param_spec_object ("file", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_MANIFEST] =
    g_param_spec_object ("manifest", NULL, NULL,
                         FOUNDRY_TYPE_FLATPAK_MANIFEST,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_flatpak_config_init (PluginFlatpakConfig *self)
{
}

PluginFlatpakConfig *
plugin_flatpak_config_new (FoundryContext         *context,
                           FoundryFlatpakManifest *manifest,
                           GFile                  *file)
{
  g_autofree char *id = NULL;
  g_autofree char *name = NULL;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST (manifest), NULL);
  g_return_val_if_fail (G_IS_FILE (file), NULL);

  name = g_file_get_basename (file);
  id = g_strdup_printf ("flatpak:%s", name);

  return g_object_new (PLUGIN_TYPE_FLATPAK_CONFIG,
                       "id", id,
                       "name", name,
                       "context", context,
                       "file", file,
                       "manifest", manifest,
                       NULL);
}

FoundryFlatpakModule *
plugin_flatpak_config_dup_primary_module (PluginFlatpakConfig *self)
{
  g_autoptr(FoundryFlatpakModules) modules = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GFile) project_dir = NULL;

  g_return_val_if_fail (PLUGIN_IS_FLATPAK_CONFIG (self), NULL);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  project_dir = foundry_context_dup_project_directory (context);

  if ((modules = foundry_flatpak_manifest_dup_modules (self->manifest)))
    {
      g_autoptr(FoundryFlatpakModule) primary = NULL;

      if (!(primary = foundry_flatpak_modules_find_primary (modules, project_dir)))
        {
          guint n_items = g_list_model_get_n_items (G_LIST_MODEL (modules));

          if (n_items > 0)
            primary = g_list_model_get_item (G_LIST_MODEL (modules), n_items - 1);
        }

      return g_steal_pointer (&primary);
    }

  return NULL;
}

char *
plugin_flatpak_config_dup_primary_module_name (PluginFlatpakConfig *self)
{
  g_autoptr(FoundryFlatpakModule) primary_module = NULL;

  g_return_val_if_fail (PLUGIN_IS_FLATPAK_CONFIG (self), NULL);

  if ((primary_module = plugin_flatpak_config_dup_primary_module (self)))
    return foundry_flatpak_module_dup_name (primary_module);

  return NULL;
}

FoundryFlatpakManifest *
plugin_flatpak_config_dup_manifest (PluginFlatpakConfig *self)
{
  g_return_val_if_fail (PLUGIN_IS_FLATPAK_CONFIG (self), NULL);

  return g_object_ref (self->manifest);
}

char *
plugin_flatpak_config_dup_id (PluginFlatpakConfig *self)
{
  g_return_val_if_fail (PLUGIN_IS_FLATPAK_CONFIG (self), NULL);

  return foundry_flatpak_manifest_dup_id (self->manifest);
}

char *
plugin_flatpak_config_dup_sdk (PluginFlatpakConfig *self)
{
  g_return_val_if_fail (PLUGIN_IS_FLATPAK_CONFIG (self), NULL);

  return foundry_flatpak_manifest_dup_sdk (self->manifest);
}

char *
plugin_flatpak_config_dup_runtime (PluginFlatpakConfig *self)
{
  g_return_val_if_fail (PLUGIN_IS_FLATPAK_CONFIG (self), NULL);

  return foundry_flatpak_manifest_dup_runtime (self->manifest);
}

char *
plugin_flatpak_config_dup_runtime_version (PluginFlatpakConfig *self)
{
  g_return_val_if_fail (PLUGIN_IS_FLATPAK_CONFIG (self), NULL);

  return foundry_flatpak_manifest_dup_runtime_version (self->manifest);
}

GFile *
plugin_flatpak_config_dup_file (PluginFlatpakConfig *self)
{
  g_return_val_if_fail (PLUGIN_IS_FLATPAK_CONFIG (self), NULL);

  return g_object_ref (self->file);
}
