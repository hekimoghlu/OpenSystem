/* foundry-build-stage.h
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

#include "foundry-build-pipeline.h"
#include "foundry-contextual.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_BUILD_STAGE (foundry_build_stage_get_type())

FOUNDRY_AVAILABLE_IN_ALL
FOUNDRY_DECLARE_INTERNAL_TYPE (FoundryBuildStage, foundry_build_stage, FOUNDRY, BUILD_STAGE, FoundryContextual)

struct _FoundryBuildStage
{
  FoundryContextual parent_instance;
};

struct _FoundryBuildStageClass
{
  FoundryContextualClass parent_class;

  FoundryBuildPipelinePhase  (*get_phase)          (FoundryBuildStage    *self);
  guint                      (*get_priority)       (FoundryBuildStage    *self);
  DexFuture                 *(*query)              (FoundryBuildStage    *self);
  DexFuture                 *(*build)              (FoundryBuildStage    *self,
                                                    FoundryBuildProgress *progress);
  DexFuture                 *(*clean)              (FoundryBuildStage    *self,
                                                    FoundryBuildProgress *progress);
  DexFuture                 *(*purge)              (FoundryBuildStage    *self,
                                                    FoundryBuildProgress *progress);
  DexFuture                 *(*find_build_flags)   (FoundryBuildStage    *self,
                                                    GFile                *file);
  DexFuture                 *(*list_build_targets) (FoundryBuildStage    *self);

  /*< private >*/
  gpointer _reserved[16];
};

FOUNDRY_AVAILABLE_IN_ALL
FoundryBuildPipeline      *foundry_build_stage_dup_pipeline       (FoundryBuildStage    *self);
FOUNDRY_AVAILABLE_IN_ALL
char                      *foundry_build_stage_dup_kind           (FoundryBuildStage    *self);
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_build_stage_set_kind           (FoundryBuildStage    *self,
                                                                   const char           *kind);
FOUNDRY_AVAILABLE_IN_ALL
FoundryBuildPipelinePhase  foundry_build_stage_get_phase          (FoundryBuildStage    *self);
FOUNDRY_AVAILABLE_IN_ALL
guint                      foundry_build_stage_get_priority       (FoundryBuildStage    *self);
FOUNDRY_AVAILABLE_IN_ALL
char                      *foundry_build_stage_dup_title          (FoundryBuildStage    *self);
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_build_stage_set_title          (FoundryBuildStage    *self,
                                                                   const char           *title);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                 *foundry_build_stage_query              (FoundryBuildStage    *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                 *foundry_build_stage_build              (FoundryBuildStage    *self,
                                                                   FoundryBuildProgress *progress);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                 *foundry_build_stage_clean              (FoundryBuildStage    *self,
                                                                   FoundryBuildProgress *progress);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                 *foundry_build_stage_purge              (FoundryBuildStage    *self,
                                                                   FoundryBuildProgress *progress);
FOUNDRY_AVAILABLE_IN_ALL
gboolean                   foundry_build_stage_get_completed      (FoundryBuildStage    *self);
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_build_stage_set_completed      (FoundryBuildStage    *self,
                                                                   gboolean              completed);
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_build_stage_invalidate         (FoundryBuildStage    *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                 *foundry_build_stage_find_build_flags   (FoundryBuildStage    *self,
                                                                   GFile                *file);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                 *foundry_build_stage_list_build_targets (FoundryBuildStage    *self);

G_END_DECLS
