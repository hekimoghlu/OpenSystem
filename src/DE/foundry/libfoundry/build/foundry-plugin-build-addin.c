/* foundry-plugin-build-addin.c
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

#include "foundry-build-pipeline.h"
#include "foundry-command.h"
#include "foundry-command-stage.h"
#include "foundry-config.h"
#include "foundry-plugin-build-addin.h"
#include "foundry-util.h"

/**
 * FoundryPluginBuildAddin:
 *
 * The [class@Foundry.PluginBuildAddin] class is a convenience object that
 * may be used by plug-ins implementing a build system.
 *
 * It allows you to specify very simple build system commands in your
 * `.plugin` definition without having to implement the class yourself.
 *
 * This is the preferred method for implementing very simple build system
 * integration within Foundry.
 */

struct _FoundryPluginBuildAddin
{
  FoundryBuildAddin  parent_instance;
  FoundryBuildStage *downloads;
  FoundryBuildStage *autogen;
  FoundryBuildStage *config;
  FoundryBuildStage *build;
  FoundryBuildStage *install;
};

G_DEFINE_FINAL_TYPE (FoundryPluginBuildAddin, foundry_plugin_build_addin, FOUNDRY_TYPE_BUILD_ADDIN)

static FoundryCommand *
create_command (FoundryContext     *context,
                const char         *command_str,
                const char * const *extra_build_opts)
{
  g_autoptr(FoundryCommand) command = NULL;
  g_autoptr(GStrvBuilder) builder = NULL;
  g_autoptr(GError) error = NULL;
  g_auto(GStrv) argv = NULL;
  int argc;

  if (foundry_str_empty0 (command_str))
    return NULL;

  if (!g_shell_parse_argv (command_str, &argc, &argv, &error))
    {
      g_critical ("Failed to parse command: \"%s\": %s",
                  command_str, error->message);
      return NULL;
    }

  builder = g_strv_builder_new ();
  g_strv_builder_addv (builder, (const char **)argv);

  if (extra_build_opts)
    g_strv_builder_addv (builder, (const char **)extra_build_opts);

  g_clear_pointer (&argv, g_strfreev);
  argv = g_strv_builder_end (builder);

  command = foundry_command_new (context);
  foundry_command_set_argv (command, (const char * const *)argv);

  return g_steal_pointer (&command);
}

static FoundryBuildStage *
foundry_plugin_build_addin_add (FoundryPluginBuildAddin   *self,
                                FoundryBuildPipeline      *pipeline,
                                const char                *command_str,
                                const char                *clean_command_str,
                                FoundryBuildPipelinePhase  phase,
                                gboolean                   phony)
{
  g_autoptr(FoundryBuildStage) stage = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryCommand) build_command = NULL;
  g_autoptr(FoundryCommand) clean_command = NULL;
  g_autoptr(GFile) srcdir = NULL;
  g_auto(GStrv) extra_build_opts = NULL;

  g_assert (FOUNDRY_IS_PLUGIN_BUILD_ADDIN (self));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));

  if (foundry_str_empty0 (command_str) &&
      foundry_str_empty0 (clean_command_str))
    return NULL;

  if (phase == FOUNDRY_BUILD_PIPELINE_PHASE_CONFIGURE)
    {
      g_autoptr(FoundryConfig) config = foundry_build_pipeline_dup_config (pipeline);
      g_auto(GStrv) config_opts = foundry_config_dup_config_opts (config);

      if (config_opts != NULL)
        extra_build_opts = g_steal_pointer (&config_opts);
    }

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  srcdir = foundry_context_dup_project_directory (context);

  build_command = create_command (context, command_str, (const char * const *)extra_build_opts);
  clean_command = create_command (context, clean_command_str, NULL);

  /* Run autogen phase from srcdir */
  if (build_command != NULL &&
      phase == FOUNDRY_BUILD_PIPELINE_PHASE_AUTOGEN &&
      g_file_is_native (srcdir))
    {
      g_autofree char *path = g_file_get_path (srcdir);
      foundry_command_set_cwd (build_command, path);
    }

  stage = foundry_command_stage_new (context, phase, build_command, clean_command, NULL, NULL, phony);
  foundry_build_pipeline_add_stage (pipeline, stage);

  return g_steal_pointer (&stage);
}

static DexFuture *
foundry_plugin_build_addin_load (FoundryBuildAddin *addin)
{
  FoundryPluginBuildAddin *self = (FoundryPluginBuildAddin *)addin;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(PeasPluginInfo) plugin_info = NULL;
  g_autofree char *build_system = NULL;

  g_assert (FOUNDRY_IS_PLUGIN_BUILD_ADDIN (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  build_system = foundry_context_dup_build_system (context);

  if ((plugin_info = foundry_build_addin_dup_plugin_info (addin)))
    {
      g_autoptr(FoundryBuildPipeline) pipeline = foundry_build_addin_dup_pipeline (addin);
      const char *x_buildsystem_name = peas_plugin_info_get_external_data (plugin_info, "BuildSystem-Name");
      const char *x_buildsystem_autogen_command = peas_plugin_info_get_external_data (plugin_info, "BuildSystem-Autogen-Command");
      const char *x_buildsystem_downloads_command = peas_plugin_info_get_external_data (plugin_info, "BuildSystem-Downloads-Command");
      const char *x_buildsystem_config_command = peas_plugin_info_get_external_data (plugin_info, "BuildSystem-Config-Command");
      const char *x_buildsystem_config_command_phony = peas_plugin_info_get_external_data (plugin_info, "BuildSystem-Config-Command-Phony");
      const char *x_buildsystem_build_command = peas_plugin_info_get_external_data (plugin_info, "BuildSystem-Build-Command");
      const char *x_buildsystem_clean_command = peas_plugin_info_get_external_data (plugin_info, "BuildSystem-Clean-Command");
      const char *x_buildsystem_install_command = peas_plugin_info_get_external_data (plugin_info, "BuildSystem-Install-Command");

      if (g_strcmp0 (build_system, x_buildsystem_name) == 0)
        {
          gboolean config_phony = FALSE;

          if (x_buildsystem_config_command_phony)
            config_phony = g_str_equal ("true", x_buildsystem_config_command_phony);

          self->downloads = foundry_plugin_build_addin_add (self, pipeline, x_buildsystem_downloads_command, NULL, FOUNDRY_BUILD_PIPELINE_PHASE_DOWNLOADS, FALSE);
          self->autogen = foundry_plugin_build_addin_add (self, pipeline, x_buildsystem_autogen_command, NULL, FOUNDRY_BUILD_PIPELINE_PHASE_AUTOGEN, FALSE);
          self->config = foundry_plugin_build_addin_add (self, pipeline, x_buildsystem_config_command, NULL, FOUNDRY_BUILD_PIPELINE_PHASE_CONFIGURE, config_phony);
          self->build = foundry_plugin_build_addin_add (self, pipeline, x_buildsystem_build_command, x_buildsystem_clean_command, FOUNDRY_BUILD_PIPELINE_PHASE_BUILD, TRUE);
          self->install = foundry_plugin_build_addin_add (self, pipeline, x_buildsystem_install_command, NULL, FOUNDRY_BUILD_PIPELINE_PHASE_INSTALL, TRUE);
        }
    }

  return dex_future_new_true ();
}

static DexFuture *
foundry_plugin_build_addin_unload (FoundryBuildAddin *addin)
{
  FoundryPluginBuildAddin *self = (FoundryPluginBuildAddin *)addin;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;

  g_assert (FOUNDRY_IS_PLUGIN_BUILD_ADDIN (self));

  pipeline = foundry_build_addin_dup_pipeline (addin);

  foundry_clear_build_stage (&self->downloads, pipeline);
  foundry_clear_build_stage (&self->autogen, pipeline);
  foundry_clear_build_stage (&self->config, pipeline);
  foundry_clear_build_stage (&self->build, pipeline);
  foundry_clear_build_stage (&self->install, pipeline);

  return dex_future_new_true ();
}

static void
foundry_plugin_build_addin_class_init (FoundryPluginBuildAddinClass *klass)
{
  FoundryBuildAddinClass *build_addin_class = FOUNDRY_BUILD_ADDIN_CLASS (klass);

  build_addin_class->load = foundry_plugin_build_addin_load;
  build_addin_class->unload = foundry_plugin_build_addin_unload;
}

static void
foundry_plugin_build_addin_init (FoundryPluginBuildAddin *self)
{
}
