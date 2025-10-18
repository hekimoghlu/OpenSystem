/* plugin-buildconfig-config.c
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

#include "plugin-buildconfig-config.h"

struct _PluginBuildconfigConfig
{
  FoundryConfig   parent_instance;
  char          **config_opts;
  char           *sdk_id;
  char           *build_system;
  char          **prebuild;
  char          **postbuild;
  char          **run_command;
  char          **build_env;
  char          **runtime_env;
  guint           can_default : 1;
};

G_DEFINE_FINAL_TYPE (PluginBuildconfigConfig, plugin_buildconfig_config, FOUNDRY_TYPE_CONFIG)

static DexFuture *
plugin_buildconfig_config_resolve_sdk (FoundryConfig *config,
                                       FoundryDevice *device)
{
  PluginBuildconfigConfig *self = (PluginBuildconfigConfig *)config;
  g_autoptr(FoundrySdkManager) sdk_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;

  g_assert (PLUGIN_IS_BUILDCONFIG_CONFIG (self));
  g_assert (FOUNDRY_IS_DEVICE (device));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  sdk_manager = foundry_context_dup_sdk_manager (context);

  if (self->sdk_id != NULL)
    return foundry_sdk_manager_find_by_id (sdk_manager, self->sdk_id);
  else
    return foundry_sdk_manager_find_by_id (sdk_manager, "host");
}

static gboolean
plugin_buildconfig_config_can_default (FoundryConfig *config,
                                       guint         *priority)
{
  *priority = 0;
  return TRUE;
}

static char **
plugin_buildconfig_config_dup_config_opts (FoundryConfig *config)
{
  return g_strdupv (PLUGIN_BUILDCONFIG_CONFIG (config)->config_opts);
}

static char *
plugin_buildconfig_config_dup_build_system (FoundryConfig *config)
{
  return g_strdup (PLUGIN_BUILDCONFIG_CONFIG (config)->build_system);
}

static FoundryCommand *
plugin_buildconfig_config_dup_default_command (FoundryConfig *config)
{
  PluginBuildconfigConfig *self = (PluginBuildconfigConfig *)config;
  g_autoptr(FoundryCommand) command = NULL;
  g_autoptr(FoundryContext) context = NULL;

  g_assert (PLUGIN_IS_BUILDCONFIG_CONFIG (self));

  if (self->run_command == NULL)
    return NULL;

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  command = foundry_command_new (context);
  foundry_command_set_argv (command, (const char * const *)self->run_command);
  foundry_command_set_environ (command, (const char * const *)self->runtime_env);

  return g_steal_pointer (&command);
}

static char **
plugin_buildconfig_config_dup_environ (FoundryConfig   *config,
                                       FoundryLocality  locality)
{
  PluginBuildconfigConfig *self = PLUGIN_BUILDCONFIG_CONFIG (config);

  switch (locality)
    {
    case FOUNDRY_LOCALITY_BUILD:
    case FOUNDRY_LOCALITY_TOOL:
      return g_strdupv (self->build_env);

    case FOUNDRY_LOCALITY_RUN:
      return g_strdupv (self->runtime_env);

    case FOUNDRY_LOCALITY_LAST:
    default:
      return NULL;
    }
}

static void
plugin_buildconfig_config_finalize (GObject *object)
{
  PluginBuildconfigConfig *self = (PluginBuildconfigConfig *)object;

  g_clear_pointer (&self->build_system, g_free);
  g_clear_pointer (&self->config_opts, g_strfreev);
  g_clear_pointer (&self->prebuild, g_strfreev);
  g_clear_pointer (&self->postbuild, g_strfreev);
  g_clear_pointer (&self->run_command, g_strfreev);
  g_clear_pointer (&self->build_env, g_strfreev);
  g_clear_pointer (&self->runtime_env, g_strfreev);
  g_clear_pointer (&self->sdk_id, g_free);

  G_OBJECT_CLASS (plugin_buildconfig_config_parent_class)->finalize (object);
}

static void
plugin_buildconfig_config_class_init (PluginBuildconfigConfigClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryConfigClass *config_class = FOUNDRY_CONFIG_CLASS (klass);

  object_class->finalize = plugin_buildconfig_config_finalize;

  config_class->can_default = plugin_buildconfig_config_can_default;
  config_class->dup_build_system = plugin_buildconfig_config_dup_build_system;
  config_class->dup_config_opts = plugin_buildconfig_config_dup_config_opts;
  config_class->dup_default_command = plugin_buildconfig_config_dup_default_command;
  config_class->dup_environ = plugin_buildconfig_config_dup_environ;
  config_class->resolve_sdk = plugin_buildconfig_config_resolve_sdk;
}

static void
plugin_buildconfig_config_init (PluginBuildconfigConfig *self)
{
  self->can_default = TRUE;
}

static char **
group_to_strv (GKeyFile   *key_file,
               const char *group)
{
  g_auto(GStrv) env = NULL;
  g_auto(GStrv) keys = NULL;
  gsize len;

  g_assert (key_file != NULL);
  g_assert (group != NULL);

  if (!g_key_file_has_group (key_file, group))
    return NULL;

  keys = g_key_file_get_keys (key_file, group, &len, NULL);

  for (gsize i = 0; i < len; i++)
    {
      g_autofree char *value = g_key_file_get_string (key_file, group, keys[i], NULL);

      if (value != NULL)
        env = g_environ_setenv (env, keys[i], value, TRUE);
    }

  return g_steal_pointer (&env);
}

static gboolean
plugin_buildconfig_config_load (PluginBuildconfigConfig *self,
                                GKeyFile                *key_file,
                                const char              *group)
{
  g_autofree char *build_env_key = NULL;
  g_autofree char *runtime_env_key = NULL;
  g_autofree char *config_opts_str = NULL;
  g_autofree char *run_command_str = NULL;
  g_autofree char *sdk_id = NULL;
  g_autofree char *prebuild_str = NULL;
  g_autofree char *postbuild_str = NULL;
  g_auto(GStrv) build_env = NULL;
  g_auto(GStrv) runtime_env = NULL;
  g_auto(GStrv) argv = NULL;
  int argc;

  g_assert (PLUGIN_IS_BUILDCONFIG_CONFIG (self));
  g_assert (key_file != NULL);
  g_assert (group != NULL);

  build_env_key = g_strdup_printf ("%s.environment", group);
  runtime_env_key = g_strdup_printf ("%s.runtime_environment", group);

  self->build_env = group_to_strv (key_file, build_env_key);
  self->runtime_env = group_to_strv (key_file, runtime_env_key);

  config_opts_str = g_key_file_get_string (key_file, group, "config-opts", NULL);
  run_command_str = g_key_file_get_string (key_file, group, "run-command", NULL);

  self->build_system = g_key_file_get_string (key_file, group, "build-system", NULL);

  if (!foundry_str_empty0 (config_opts_str))
    {
      g_auto(GStrv) config_opts = NULL;

      if (!g_shell_parse_argv (config_opts_str, &argc, &config_opts, NULL))
        return FALSE;

      self->config_opts = g_steal_pointer (&config_opts);
    }

  if (!foundry_str_empty0 (run_command_str))
    {
      g_auto(GStrv) run_command = NULL;

      if (!g_shell_parse_argv (run_command_str, &argc, &run_command, NULL))
        return FALSE;

      self->run_command = g_steal_pointer (&run_command);
    }

  if (!g_key_file_get_boolean (key_file, group, "default", NULL))
    self->can_default = FALSE;

  if ((sdk_id = g_key_file_get_string (key_file, group, "runtime", NULL)))
    {
      /* Special case what we have previously supported in Builder.
       * That way we can open them without additional tweaking.
       */
      if (g_str_has_prefix (sdk_id, "flatpak:"))
        self->sdk_id = g_strdup (sdk_id + strlen ("flatpak:"));
      else if (g_str_has_prefix (sdk_id, "podman:"))
        self->sdk_id = g_strdup (sdk_id + strlen ("podman:"));
      else
        self->sdk_id = g_steal_pointer (&sdk_id);
    }

  if ((prebuild_str = g_key_file_get_string (key_file, group, "prebuild", NULL)) &&
      g_shell_parse_argv (prebuild_str, &argc, &argv, NULL))
    self->prebuild = g_steal_pointer (&argv);

  if ((postbuild_str = g_key_file_get_string (key_file, group, "postbuild", NULL)) &&
      g_shell_parse_argv (postbuild_str, &argc, &argv, NULL))
    self->postbuild = g_steal_pointer (&argv);

  return TRUE;
}

FoundryConfig *
plugin_buildconfig_config_new (FoundryContext *context,
                               GKeyFile       *key_file,
                               const char     *group)
{
  g_autoptr(PluginBuildconfigConfig) self = NULL;
  g_autofree char *id = NULL;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (key_file != NULL, NULL);
  g_return_val_if_fail (group != NULL, NULL);

  id = g_strdup_printf ("buildconfig:%s", group);

  self = g_object_new (PLUGIN_TYPE_BUILDCONFIG_CONFIG,
                       "id", id,
                       "context", context,
                       NULL);

  if (!plugin_buildconfig_config_load (self, key_file, group))
    return NULL;

  return FOUNDRY_CONFIG (g_steal_pointer (&self));
}

char **
plugin_buildconfig_config_dup_prebuild (PluginBuildconfigConfig *self)
{
  g_return_val_if_fail (PLUGIN_IS_BUILDCONFIG_CONFIG (self), NULL);

  return g_strdupv (self->prebuild);
}

char **
plugin_buildconfig_config_dup_postbuild (PluginBuildconfigConfig *self)
{
  g_return_val_if_fail (PLUGIN_IS_BUILDCONFIG_CONFIG (self), NULL);

  return g_strdupv (self->postbuild);
}
