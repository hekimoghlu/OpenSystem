/* foundry-build-pipeline.h
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

#pragma once

#include "foundry-contextual.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_BUILD_PIPELINE       (foundry_build_pipeline_get_type())
#define FOUNDRY_TYPE_BUILD_PIPELINE_PHASE (foundry_build_pipeline_phase_get_type())

#define FOUNDRY_BUILD_PIPELINE_PHASE_MASK(p)        ((p) & ((1<<11)-1))
#define FOUNDRY_BUILD_PIPELINE_PHASE_WHENCE_MASK(p) ((p) & (FOUNDRY_BUILD_PIPELINE_PHASE_BEFORE|FOUNDRY_BUILD_PIPELINE_PHASE_AFTER))

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryBuildPipeline, foundry_build_pipeline, FOUNDRY, BUILD_PIPELINE, FoundryContextual)

FOUNDRY_AVAILABLE_IN_ALL
GType                 foundry_build_pipeline_phase_get_type      (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture            *foundry_build_pipeline_new                 (FoundryContext            *context,
                                                                  FoundryConfig             *config,
                                                                  FoundryDevice             *device,
                                                                  FoundrySdk                *sdk,
                                                                  gboolean                   enable_adddins) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
FoundryBuildProgress *foundry_build_pipeline_build               (FoundryBuildPipeline      *self,
                                                                  FoundryBuildPipelinePhase  phase,
                                                                  int                        pty_fd,
                                                                  DexCancellable            *cancellable);
FOUNDRY_AVAILABLE_IN_ALL
FoundryBuildProgress *foundry_build_pipeline_clean               (FoundryBuildPipeline      *self,
                                                                  FoundryBuildPipelinePhase  phase,
                                                                  int                        pty_fd,
                                                                  DexCancellable            *cancellable);
FOUNDRY_AVAILABLE_IN_ALL
FoundryBuildProgress *foundry_build_pipeline_purge               (FoundryBuildPipeline      *self,
                                                                  FoundryBuildPipelinePhase  phase,
                                                                  int                        pty_fd,
                                                                  DexCancellable            *cancellable);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTriplet       *foundry_build_pipeline_dup_triplet         (FoundryBuildPipeline      *self);
FOUNDRY_AVAILABLE_IN_ALL
char                 *foundry_build_pipeline_dup_arch            (FoundryBuildPipeline      *self);
FOUNDRY_AVAILABLE_IN_ALL
char                 *foundry_build_pipeline_dup_builddir        (FoundryBuildPipeline      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryConfig        *foundry_build_pipeline_dup_config          (FoundryBuildPipeline      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDevice        *foundry_build_pipeline_dup_device          (FoundryBuildPipeline      *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundrySdk           *foundry_build_pipeline_dup_sdk             (FoundryBuildPipeline      *self);
FOUNDRY_AVAILABLE_IN_ALL
void                  foundry_build_pipeline_add_stage           (FoundryBuildPipeline      *self,
                                                                  FoundryBuildStage         *stage);
FOUNDRY_AVAILABLE_IN_ALL
void                  foundry_build_pipeline_remove_stage        (FoundryBuildPipeline      *self,
                                                                  FoundryBuildStage         *stage);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture            *foundry_build_pipeline_prepare             (FoundryBuildPipeline      *self,
                                                                  FoundryProcessLauncher    *launcher,
                                                                  FoundryBuildPipelinePhase  phase) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture            *foundry_build_pipeline_prepare_for_run     (FoundryBuildPipeline      *self,
                                                                  FoundryProcessLauncher    *launcher) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture            *foundry_build_pipeline_contains_program    (FoundryBuildPipeline      *self,
                                                                  const char                *program) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture            *foundry_build_pipeline_find_build_flags    (FoundryBuildPipeline      *self,
                                                                  GFile                     *file) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture            *foundry_build_pipeline_list_build_targets  (FoundryBuildPipeline      *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_1_1
void                  foundry_build_pipeline_setenv              (FoundryBuildPipeline      *self,
                                                                  const char                *variable,
                                                                  const char                *value);

#ifndef __GI_SCANNER__
static inline void
foundry_clear_build_stage (FoundryBuildStage    **stageptr,
                           FoundryBuildPipeline  *pipeline)
{
  if (*stageptr != NULL)
    {
      if (pipeline != NULL)
        foundry_build_pipeline_remove_stage (pipeline, *stageptr);
      g_clear_object (stageptr);
    }
}
#endif

G_END_DECLS
