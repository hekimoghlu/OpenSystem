/* foundry-build-progress.h
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

#define FOUNDRY_TYPE_BUILD_PROGRESS (foundry_build_progress_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryBuildProgress, foundry_build_progress, FOUNDRY, BUILD_PROGRESS, FoundryContextual)

FOUNDRY_AVAILABLE_IN_ALL
FoundryBuildPipelinePhase  foundry_build_progress_get_phase       (FoundryBuildProgress   *self);
FOUNDRY_AVAILABLE_IN_ALL
DexCancellable            *foundry_build_progress_dup_cancellable (FoundryBuildProgress   *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture                 *foundry_build_progress_await           (FoundryBuildProgress   *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_build_progress_print           (FoundryBuildProgress   *self,
                                                                   const char             *format,
                                                                   ...) G_GNUC_PRINTF (2, 3);
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_build_progress_setup_pty       (FoundryBuildProgress   *self,
                                                                   FoundryProcessLauncher *launcher);
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_build_progress_add_artifact    (FoundryBuildProgress   *self,
                                                                   GFile                  *file);
FOUNDRY_AVAILABLE_IN_ALL
GListModel                *foundry_build_progress_list_artifacts  (FoundryBuildProgress   *self);

G_END_DECLS
