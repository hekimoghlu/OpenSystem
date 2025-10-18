/* plugin-sarif-build-stage.c
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

#include "plugin-sarif-build-stage.h"
#include "plugin-sarif-service.h"

struct _PluginSarifBuildStage
{
  FoundryBuildStage parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginSarifBuildStage, plugin_sarif_build_stage, FOUNDRY_TYPE_BUILD_STAGE)

static FoundryBuildPipelinePhase
plugin_sarif_build_stage_get_phase (FoundryBuildStage *stage)
{
  return FOUNDRY_BUILD_PIPELINE_PHASE_BUILD | FOUNDRY_BUILD_PIPELINE_PHASE_BEFORE;
}

static DexFuture *
plugin_sarif_build_stage_query (FoundryBuildStage *stage)
{
  foundry_build_stage_set_completed (stage, FALSE);
  return dex_future_new_true ();
}

static DexFuture *
plugin_sarif_build_stage_clear (FoundryBuildStage    *stage,
                                FoundryBuildProgress *progress)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(PluginSarifService) service = NULL;
  g_autoptr(FoundryContext) context = NULL;

  g_assert (PLUGIN_IS_SARIF_BUILD_STAGE (stage));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (stage));
  service = foundry_context_dup_service_typed (context, PLUGIN_TYPE_SARIF_SERVICE);

  /* Notify the build service of the builddir just in case it has changed
   * for this pipeline (and before it starts getting SARIF output).
   */
  if ((pipeline = foundry_build_stage_dup_pipeline (stage)))
    {
      g_autofree char *builddir = foundry_build_pipeline_dup_builddir (pipeline);

      plugin_sarif_service_set_builddir (service, builddir);
    }

  plugin_sarif_service_reset (service);

  return dex_future_new_true ();
}

static void
plugin_sarif_build_stage_class_init (PluginSarifBuildStageClass *klass)
{
  FoundryBuildStageClass *build_stage_class = FOUNDRY_BUILD_STAGE_CLASS (klass);

  build_stage_class->get_phase = plugin_sarif_build_stage_get_phase;
  build_stage_class->query = plugin_sarif_build_stage_query;
  build_stage_class->build = plugin_sarif_build_stage_clear;
  build_stage_class->clean = plugin_sarif_build_stage_clear;
  build_stage_class->purge = plugin_sarif_build_stage_clear;
}

static void
plugin_sarif_build_stage_init (PluginSarifBuildStage *self)
{
}
