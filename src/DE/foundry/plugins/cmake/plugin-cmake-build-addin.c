/* plugin-cmake-build-addin.c
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

#include <glib/gi18n-lib.h>

#include "plugin-cmake-build-addin.h"
#include "plugin-cmake-prepare-stage.h"

struct _PluginCmakeBuildAddin
{
  FoundryBuildAddin  parent_instance;
  FoundryBuildStage *build;
  FoundryBuildStage *config;
  FoundryBuildStage *install;
};

G_DEFINE_FINAL_TYPE (PluginCmakeBuildAddin, plugin_cmake_build_addin, FOUNDRY_TYPE_BUILD_ADDIN)

static gboolean
contains_option (char       **options,
                 const char  *option)
{
  if (options == NULL)
    return FALSE;

  for (guint i = 0; options[i]; i++)
    {
      if (g_str_has_prefix (options[i], option))
        return TRUE;
    }

  return FALSE;
}

static DexFuture *
plugin_cmake_build_addin_load (FoundryBuildAddin *build_addin)
{
  PluginCmakeBuildAddin *self = (PluginCmakeBuildAddin *)build_addin;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildStage) prepare_stage = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autoptr(FoundrySdk) sdk = NULL;
  g_autoptr(FoundryCommand) build_command = NULL;
  g_autoptr(FoundryCommand) clean_command = NULL;
  g_autoptr(FoundryCommand) config_command = NULL;
  g_autoptr(FoundryCommand) install_command = NULL;
  g_autoptr(GFile) project_dir = NULL;
  g_autofree char *build_system = NULL;
  g_autofree char *prefix = NULL;
  g_autofree char *libdir = NULL;
  g_autofree char *builddir = NULL;
  g_auto(GStrv) config_opts = NULL;

  g_assert (PLUGIN_IS_CMAKE_BUILD_ADDIN (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  project_dir = foundry_context_dup_project_directory (context);
  build_system = foundry_context_dup_build_system (context);

  if (!g_file_is_native (project_dir) ||
      !(foundry_str_equal0 (build_system, "cmake") ||
        foundry_str_equal0 (build_system, "cmake-ninja")))
    return dex_future_new_true ();

  pipeline = foundry_build_addin_dup_pipeline (build_addin);
  config = foundry_build_pipeline_dup_config (pipeline);
  sdk = foundry_build_pipeline_dup_sdk (pipeline);

  prefix = foundry_sdk_dup_config_option (sdk, FOUNDRY_SDK_CONFIG_OPTION_PREFIX);
  libdir = foundry_sdk_dup_config_option (sdk, FOUNDRY_SDK_CONFIG_OPTION_LIBDIR);
  config_opts = foundry_config_dup_config_opts (config);
  builddir = foundry_build_pipeline_dup_builddir (pipeline);

  config_command = foundry_command_new (context);
  build_command = foundry_command_new (context);
  clean_command = foundry_command_new (context);
  install_command = foundry_command_new (context);

  foundry_command_set_cwd (config_command, builddir);
  foundry_command_set_cwd (build_command, builddir);
  foundry_command_set_cwd (clean_command, builddir);
  foundry_command_set_cwd (install_command, builddir);

  /* DESTDIR= will get set anyway so use something typical */
  if (prefix == NULL)
    prefix = g_strdup ("/usr");

  /* TODO: Determine CMakeLists.txt location instead of expecting it
   * at the root of the project. We might defer this to FoundryConfig
   * to point us in the right location.
   */

  /* Setup prepare stage for queries */
  prepare_stage = plugin_cmake_prepare_stage_new (context, builddir);
  foundry_build_pipeline_add_stage (pipeline, prepare_stage);

  /* Setup config stage */
    {
      g_autoptr(GStrvBuilder) builder = g_strv_builder_new ();
      g_autoptr(FoundryBuildStage) stage = NULL;
      g_autoptr(GFile) query_file = NULL;
      g_auto(GStrv) argv = NULL;

      g_strv_builder_add (builder, "cmake");
      g_strv_builder_add (builder, "-G");
      g_strv_builder_add (builder, "Ninja");
      g_strv_builder_add (builder, ".");
      g_strv_builder_add (builder, g_file_peek_path (project_dir));
      g_strv_builder_add (builder, "-DCMAKE_EXPORT_COMPILE_COMMANDS=1");

      if (!contains_option (config_opts, "-DCMAKE_BUILD_TYPE="))
        g_strv_builder_add (builder, "-DCMAKE_BUILD_TYPE=RelWithDebInfo");

      if (!contains_option (config_opts, "-DCMAKE_INSTALL_PREFIX="))
        {
          g_autofree char *prefix_arg = g_strdup_printf ("-DCMAKE_INSTALL_PREFIX=%s", prefix);
          g_strv_builder_add (builder, prefix_arg);
        }

      if (config_opts != NULL)
        g_strv_builder_addv (builder, (const char **)config_opts);

      argv = g_strv_builder_end (builder);
      foundry_command_set_argv (config_command, (const char * const *)argv);

      query_file = g_file_new_build_filename (builddir, "build.ninja", NULL);

      stage = foundry_command_stage_new (context,
                                         FOUNDRY_BUILD_PIPELINE_PHASE_CONFIGURE,
                                         config_command, NULL, NULL, query_file, FALSE);
      foundry_build_stage_set_kind (stage, "cmake");
      foundry_build_stage_set_title (stage, _("Configure CMake Project"));
      foundry_build_pipeline_add_stage (pipeline, stage);
    }

  /* Setup build stage */
    {
      g_autoptr(FoundryBuildStage) stage = NULL;

      foundry_command_set_argv (build_command, FOUNDRY_STRV_INIT ("ninja"));
      foundry_command_set_argv (clean_command, FOUNDRY_STRV_INIT ("ninja", "clean"));

      stage = foundry_command_stage_new (context,
                                         FOUNDRY_BUILD_PIPELINE_PHASE_BUILD,
                                         build_command, clean_command, NULL, NULL, TRUE);
      foundry_build_stage_set_kind (stage, "cmake");
      foundry_build_stage_set_title (stage, _("Build CMake Project"));
      foundry_build_pipeline_add_stage (pipeline, stage);
    }

  /* Setup install stage */
    {
      g_autoptr(FoundryBuildStage) stage = NULL;

      foundry_command_set_argv (install_command, FOUNDRY_STRV_INIT ("ninja", "install"));

      stage = foundry_command_stage_new (context,
                                         FOUNDRY_BUILD_PIPELINE_PHASE_INSTALL,
                                         install_command, NULL, NULL, NULL, TRUE);
      foundry_build_stage_set_kind (stage, "cmake");
      foundry_build_stage_set_title (stage, _("Install CMake Project"));
      foundry_build_pipeline_add_stage (pipeline, stage);
    }

  return dex_future_new_true ();
}

static DexFuture *
plugin_cmake_build_addin_unload (FoundryBuildAddin *build_addin)
{
  PluginCmakeBuildAddin *self = (PluginCmakeBuildAddin *)build_addin;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;

  g_assert (PLUGIN_IS_CMAKE_BUILD_ADDIN (self));

  pipeline = foundry_build_addin_dup_pipeline (build_addin);

  foundry_clear_build_stage (&self->build, pipeline);
  foundry_clear_build_stage (&self->config, pipeline);
  foundry_clear_build_stage (&self->install, pipeline);

  return dex_future_new_true ();
}

static void
plugin_cmake_build_addin_finalize (GObject *object)
{
  G_OBJECT_CLASS (plugin_cmake_build_addin_parent_class)->finalize (object);
}

static void
plugin_cmake_build_addin_class_init (PluginCmakeBuildAddinClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryBuildAddinClass *build_addin_class = FOUNDRY_BUILD_ADDIN_CLASS (klass);

  object_class->finalize = plugin_cmake_build_addin_finalize;

  build_addin_class->load = plugin_cmake_build_addin_load;
  build_addin_class->unload = plugin_cmake_build_addin_unload;
}

static void
plugin_cmake_build_addin_init (PluginCmakeBuildAddin *self)
{
}
