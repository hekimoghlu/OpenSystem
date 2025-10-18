/* foundry-command-stage.h
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

#include "foundry-build-stage.h"
#include "foundry-command.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_COMMAND_STAGE (foundry_command_stage_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryCommandStage, foundry_command_stage, FOUNDRY, COMMAND_STAGE, FoundryBuildStage)

FOUNDRY_AVAILABLE_IN_ALL
FoundryBuildStage *foundry_command_stage_new               (FoundryContext            *context,
                                                            FoundryBuildPipelinePhase  phase,
                                                            FoundryCommand            *build_command,
                                                            FoundryCommand            *clean_command,
                                                            FoundryCommand            *purge_command,
                                                            GFile                     *query_file,
                                                            gboolean                   phony);
FOUNDRY_AVAILABLE_IN_ALL
FoundryCommand    *foundry_command_stage_dup_build_command (FoundryCommandStage       *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryCommand    *foundry_command_stage_dup_clean_command (FoundryCommandStage       *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryCommand    *foundry_command_stage_dup_purge_command (FoundryCommandStage       *self);
FOUNDRY_AVAILABLE_IN_ALL
GFile             *foundry_command_stage_dup_query_file    (FoundryCommandStage       *self);

G_END_DECLS
