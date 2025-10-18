/* foundry-linked-pipeline-stage.h
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

#pragma once

#include "foundry-build-stage.h"
#include "foundry-build-pipeline.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LINKED_PIPELINE_STAGE (foundry_linked_pipeline_stage_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryLinkedPipelineStage, foundry_linked_pipeline_stage, FOUNDRY, LINKED_PIPELINE_STAGE, FoundryBuildStage)

FOUNDRY_AVAILABLE_IN_ALL
FoundryBuildStage *foundry_linked_pipeline_stage_new (FoundryContext            *context,
                                                      FoundryBuildPipeline      *linked_pipeline,
                                                      FoundryBuildPipelinePhase  target);

G_END_DECLS
