/* foundry-sdk.h
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
#include "foundry-process-launcher.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

typedef enum _FoundrySdkConfigOption
{
  FOUNDRY_SDK_CONFIG_OPTION_PREFIX = 1,
  FOUNDRY_SDK_CONFIG_OPTION_LIBDIR,
} FoundrySdkConfigOption;

#define FOUNDRY_TYPE_SDK (foundry_sdk_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundrySdk, foundry_sdk, FOUNDRY, SDK, FoundryContextual)

struct _FoundrySdkClass
{
  FoundryContextualClass parent_class;

  DexFuture  *(*prepare_to_build)   (FoundrySdk                 *self,
                                     FoundryBuildPipeline       *pipeline,
                                     FoundryProcessLauncher     *launcher,
                                     FoundryBuildPipelinePhase   phase);
  DexFuture  *(*prepare_to_run)     (FoundrySdk                 *self,
                                     FoundryBuildPipeline       *pipeline,
                                     FoundryProcessLauncher     *launcher);
  DexFuture  *(*contains_program)   (FoundrySdk                 *self,
                                     const char                 *program);
  DexFuture  *(*install)            (FoundrySdk                 *self,
                                     FoundryOperation           *operation,
                                     DexCancellable             *cancellable);
  char       *(*dup_config_option)  (FoundrySdk                 *self,
                                     FoundrySdkConfigOption      option);

  /*< private >*/
  gpointer _reserved[11];
};

FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_sdk_get_active         (FoundrySdk                *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_sdk_dup_id             (FoundrySdk                *self);
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_sdk_set_id             (FoundrySdk                *self,
                                           const char                *id);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_sdk_dup_kind           (FoundrySdk                *self);
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_sdk_set_kind           (FoundrySdk                *self,
                                           const char                *kind);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_sdk_dup_name           (FoundrySdk                *self);
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_sdk_set_name           (FoundrySdk                *self,
                                           const char                *name);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_sdk_dup_arch           (FoundrySdk                *self);
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_sdk_set_arch           (FoundrySdk                *self,
                                           const char                *arch);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_sdk_get_installed      (FoundrySdk                *self);
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_sdk_set_installed      (FoundrySdk                *self,
                                           gboolean                   installed);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_sdk_get_extension_only (FoundrySdk                *self);
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_sdk_set_extension_only (FoundrySdk                *self,
                                           gboolean                   extension_only);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_sdk_prepare_to_build   (FoundrySdk                *self,
                                           FoundryBuildPipeline      *pipeline,
                                           FoundryProcessLauncher    *launcher,
                                           FoundryBuildPipelinePhase  phase);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_sdk_prepare_to_run     (FoundrySdk                *self,
                                           FoundryBuildPipeline      *pipeline,
                                           FoundryProcessLauncher    *launcher);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_sdk_contains_program   (FoundrySdk                *self,
                                           const char                *program);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_sdk_discover_shell     (FoundrySdk                *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_sdk_install            (FoundrySdk                *self,
                                           FoundryOperation          *operation,
                                           DexCancellable            *cancellable);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_sdk_dup_config_option  (FoundrySdk                *self,
                                           FoundrySdkConfigOption     option);
FOUNDRY_AVAILABLE_IN_1_1
DexFuture *foundry_sdk_build_simple       (FoundrySdk                *self,
                                           FoundryBuildPipeline      *pipeline,
                                           const char * const        *argv);

G_END_DECLS
