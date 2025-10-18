/* foundry-build-manager.h
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

#include <libdex.h>

#include "foundry-service.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_BUILD_MANAGER (foundry_build_manager_get_type())
#define FOUNDRY_BUILD_ERROR        (foundry_build_error_quark())

typedef enum _FoundryBuildError
{
  FOUNDRY_BUILD_ERROR_UNKNOWN = 0,
  FOUNDRY_BUILD_ERROR_INVALID_CONFIG,
  FOUNDRY_BUILD_ERROR_INVALID_DEVICE,
  FOUNDRY_BUILD_ERROR_INVALID_SDK,
} FoundryBuildError;

FOUNDRY_AVAILABLE_IN_ALL
FOUNDRY_DECLARE_INTERNAL_TYPE (FoundryBuildManager, foundry_build_manager, FOUNDRY, BUILD_MANAGER, FoundryService)

FOUNDRY_AVAILABLE_IN_ALL
GQuark     foundry_build_error_quark             (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_build_manager_invalidate      (FoundryBuildManager *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_build_manager_load_pipeline   (FoundryBuildManager *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
int        foundry_build_manager_get_default_pty (FoundryBuildManager *self);
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_build_manager_set_default_pty (FoundryBuildManager *self,
                                                  int                  pty_fd);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_build_manager_build           (FoundryBuildManager *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_build_manager_clean           (FoundryBuildManager *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_build_manager_purge           (FoundryBuildManager *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_build_manager_rebuild         (FoundryBuildManager *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_build_manager_stop            (FoundryBuildManager *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_build_manager_get_busy        (FoundryBuildManager *self);

G_END_DECLS
