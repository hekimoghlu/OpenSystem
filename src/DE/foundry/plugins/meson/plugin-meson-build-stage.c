/* plugin-meson-build-stage.c
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

#include "plugin-meson-build-stage.h"

struct _PluginMesonBuildStage
{
  FoundryBuildStage parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginMesonBuildStage, plugin_meson_build_stage, PLUGIN_TYPE_MESON_BASE_STAGE)

static DexFuture *
plugin_meson_build_stage_run_fiber (PluginMesonBuildStage *self,
                                    FoundryBuildProgress  *progress,
                                    FoundryBuildPipeline  *pipeline,
                                    const char            *command)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *builddir = NULL;
  g_autofree char *ninja = NULL;

  g_assert (PLUGIN_IS_MESON_BUILD_STAGE (self));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (command != NULL);

  builddir = plugin_meson_base_stage_dup_builddir (PLUGIN_MESON_BASE_STAGE (self));
  ninja = plugin_meson_base_stage_dup_ninja (PLUGIN_MESON_BASE_STAGE (self));
  cancellable = foundry_build_progress_dup_cancellable (progress);

  launcher = foundry_process_launcher_new ();

  if (!dex_await (foundry_build_pipeline_prepare (pipeline, launcher, FOUNDRY_BUILD_PIPELINE_PHASE_BUILD), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  foundry_process_launcher_set_cwd (launcher, builddir);
  foundry_process_launcher_append_argv (launcher, ninja);
  foundry_process_launcher_append_argv (launcher, command);

  foundry_build_progress_setup_pty (progress, launcher);

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return foundry_subprocess_wait_check (subprocess, cancellable);
}

static DexFuture *
plugin_meson_build_stage_build (FoundryBuildStage    *build_stage,
                                FoundryBuildProgress *progress)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;

  g_assert (PLUGIN_IS_MESON_BUILD_STAGE (build_stage));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));

  pipeline = foundry_build_stage_dup_pipeline (build_stage);

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_meson_build_stage_run_fiber),
                                  4,
                                  FOUNDRY_TYPE_BUILD_STAGE, build_stage,
                                  FOUNDRY_TYPE_BUILD_PROGRESS, progress,
                                  FOUNDRY_TYPE_BUILD_PIPELINE, pipeline,
                                  G_TYPE_STRING, "all");
}

static DexFuture *
plugin_meson_build_stage_clean (FoundryBuildStage    *build_stage,
                                FoundryBuildProgress *progress)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;

  g_assert (PLUGIN_IS_MESON_BUILD_STAGE (build_stage));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));

  pipeline = foundry_build_stage_dup_pipeline (build_stage);

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_meson_build_stage_run_fiber),
                                  4,
                                  FOUNDRY_TYPE_BUILD_STAGE, build_stage,
                                  FOUNDRY_TYPE_BUILD_PROGRESS, progress,
                                  FOUNDRY_TYPE_BUILD_PIPELINE, pipeline,
                                  G_TYPE_STRING, "clean");
}

static DexFuture *
plugin_meson_build_stage_query (FoundryBuildStage *build_stage)
{
  foundry_build_stage_set_completed (build_stage, FALSE);
  return dex_future_new_true ();
}

static FoundryBuildPipelinePhase
plugin_meson_build_stage_get_phase (FoundryBuildStage *build_stage)
{
  return FOUNDRY_BUILD_PIPELINE_PHASE_BUILD;
}

static void
plugin_meson_build_stage_class_init (PluginMesonBuildStageClass *klass)
{
  FoundryBuildStageClass *build_stage_class = FOUNDRY_BUILD_STAGE_CLASS (klass);

  build_stage_class->build = plugin_meson_build_stage_build;
  build_stage_class->clean = plugin_meson_build_stage_clean;
  build_stage_class->get_phase = plugin_meson_build_stage_get_phase;
  build_stage_class->query = plugin_meson_build_stage_query;
}

static void
plugin_meson_build_stage_init (PluginMesonBuildStage *self)
{
}
