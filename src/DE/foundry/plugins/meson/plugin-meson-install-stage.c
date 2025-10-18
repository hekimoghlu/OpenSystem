/* plugin-meson-install-stage.c
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

#include "plugin-meson-install-stage.h"

struct _PluginMesonInstallStage
{
  FoundryBuildStage parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginMesonInstallStage, plugin_meson_install_stage, PLUGIN_TYPE_MESON_BASE_STAGE)

static DexFuture *
plugin_meson_install_stage_run_fiber (PluginMesonInstallStage *self,
                                      FoundryBuildProgress    *progress,
                                      FoundryBuildPipeline    *pipeline)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *builddir = NULL;
  g_autofree char *meson = NULL;

  g_assert (PLUGIN_IS_MESON_INSTALL_STAGE (self));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));

  builddir = plugin_meson_base_stage_dup_builddir (PLUGIN_MESON_BASE_STAGE (self));
  meson = plugin_meson_base_stage_dup_meson (PLUGIN_MESON_BASE_STAGE (self));
  cancellable = foundry_build_progress_dup_cancellable (progress);

  launcher = foundry_process_launcher_new ();

  if (!dex_await (foundry_build_pipeline_prepare (pipeline, launcher, FOUNDRY_BUILD_PIPELINE_PHASE_BUILD), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  foundry_process_launcher_set_cwd (launcher, builddir);
  foundry_process_launcher_append_argv (launcher, meson);
  foundry_process_launcher_append_argv (launcher, "install");
  foundry_process_launcher_append_argv (launcher, "--no-rebuild");

  foundry_build_progress_setup_pty (progress, launcher);

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return foundry_subprocess_wait_check (subprocess, cancellable);
}

static DexFuture *
plugin_meson_install_stage_build (FoundryBuildStage    *build_stage,
                                  FoundryBuildProgress *progress)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;

  g_assert (PLUGIN_IS_MESON_INSTALL_STAGE (build_stage));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));

  pipeline = foundry_build_stage_dup_pipeline (build_stage);

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_meson_install_stage_run_fiber),
                                  3,
                                  PLUGIN_TYPE_MESON_INSTALL_STAGE, build_stage,
                                  FOUNDRY_TYPE_BUILD_PROGRESS, progress,
                                  FOUNDRY_TYPE_BUILD_PIPELINE, pipeline);
}

static DexFuture *
plugin_meson_install_stage_query (FoundryBuildStage *build_stage)
{
  g_assert (PLUGIN_IS_MESON_INSTALL_STAGE (build_stage));

  foundry_build_stage_set_completed (build_stage, FALSE);

  return dex_future_new_true ();
}

static FoundryBuildPipelinePhase
plugin_meson_install_stage_get_phase (FoundryBuildStage *build_stage)
{
  return FOUNDRY_BUILD_PIPELINE_PHASE_INSTALL;
}

static void
plugin_meson_install_stage_class_init (PluginMesonInstallStageClass *klass)
{
  FoundryBuildStageClass *build_stage_class = FOUNDRY_BUILD_STAGE_CLASS (klass);

  build_stage_class->build = plugin_meson_install_stage_build;
  build_stage_class->query = plugin_meson_install_stage_query;
  build_stage_class->get_phase = plugin_meson_install_stage_get_phase;
}

static void
plugin_meson_install_stage_init (PluginMesonInstallStage *self)
{
}
