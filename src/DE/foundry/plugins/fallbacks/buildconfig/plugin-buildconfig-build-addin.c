/* plugin-buildconfig-build-addin.c
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

#include "plugin-buildconfig-build-addin.h"
#include "plugin-buildconfig-config.h"

struct _PluginBuildconfigBuildAddin
{
  FoundryBuildAddin  parent_instance;
  FoundryBuildStage *prebuild;
  FoundryBuildStage *postbuild;
};

G_DEFINE_FINAL_TYPE (PluginBuildconfigBuildAddin, plugin_buildconfig_build_addin, FOUNDRY_TYPE_BUILD_ADDIN)

static FoundryBuildStage *
create_stage (FoundryContext            *context,
              const char * const        *argv,
              FoundryBuildPipelinePhase  phase)
{
  g_autoptr(FoundryCommand) command = NULL;
  g_autoptr(FoundryCommandStage) stage = NULL;
  g_autoptr(GFile) project_dir = NULL;
  g_auto(GStrv) environ = NULL;

  if (context == NULL || argv == NULL || argv[0] == NULL)
    return NULL;

  command = foundry_command_new (context);
  foundry_command_set_argv (command, argv);

  project_dir = foundry_context_dup_project_directory (context);

  if (g_file_is_native (project_dir))
    environ = g_environ_setenv (environ, "SRCDIR", g_file_peek_path (project_dir), TRUE);

  if (environ != NULL)
    foundry_command_set_environ (command, (const char * const *)environ);

  return foundry_command_stage_new (context, phase, command, NULL, NULL, NULL, FALSE);
}

static DexFuture *
plugin_buildconfig_build_addin_load (FoundryBuildAddin *build_addin)
{
  PluginBuildconfigBuildAddin *self = (PluginBuildconfigBuildAddin *)build_addin;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryConfig) config = NULL;

  g_assert (PLUGIN_IS_BUILDCONFIG_BUILD_ADDIN (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  pipeline = foundry_build_addin_dup_pipeline (build_addin);
  config = foundry_build_pipeline_dup_config (pipeline);

  if (PLUGIN_IS_BUILDCONFIG_CONFIG (config))
    {
      PluginBuildconfigConfig *buildconfig = PLUGIN_BUILDCONFIG_CONFIG (config);
      g_auto(GStrv) prebuild = plugin_buildconfig_config_dup_prebuild (buildconfig);
      g_auto(GStrv) postbuild = plugin_buildconfig_config_dup_postbuild (buildconfig);
      g_autoptr(FoundryBuildStage) prebuild_stage = NULL;
      g_autoptr(FoundryBuildStage) postbuild_stage = NULL;

      if ((prebuild_stage = create_stage (context,
                                          (const char * const *)prebuild,
                                          FOUNDRY_BUILD_PIPELINE_PHASE_BUILD|FOUNDRY_BUILD_PIPELINE_PHASE_BEFORE)))
        foundry_build_pipeline_add_stage (pipeline, prebuild_stage);

      if ((postbuild_stage = create_stage (context,
                                           (const char * const *)postbuild,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_BUILD|FOUNDRY_BUILD_PIPELINE_PHASE_AFTER)))
        foundry_build_pipeline_add_stage (pipeline, postbuild_stage);
    }


  return dex_future_new_true ();
}

static DexFuture *
plugin_buildconfig_build_addin_unload (FoundryBuildAddin *build_addin)
{
  PluginBuildconfigBuildAddin *self = (PluginBuildconfigBuildAddin *)build_addin;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;

  g_assert (PLUGIN_IS_BUILDCONFIG_BUILD_ADDIN (self));

  pipeline = foundry_build_addin_dup_pipeline (build_addin);

  foundry_clear_build_stage (&self->prebuild, pipeline);
  foundry_clear_build_stage (&self->postbuild, pipeline);

  return dex_future_new_true ();
}

static void
plugin_buildconfig_build_addin_class_init (PluginBuildconfigBuildAddinClass *klass)
{
  FoundryBuildAddinClass *build_addin_class = FOUNDRY_BUILD_ADDIN_CLASS (klass);

  build_addin_class->load = plugin_buildconfig_build_addin_load;
  build_addin_class->unload = plugin_buildconfig_build_addin_unload;
}

static void
plugin_buildconfig_build_addin_init (PluginBuildconfigBuildAddin *self)
{
}
