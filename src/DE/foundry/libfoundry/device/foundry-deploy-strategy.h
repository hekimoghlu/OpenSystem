/* foundry-deploy-strategy.h
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

#include <libpeas.h>

#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEPLOY_STRATEGY (foundry_deploy_strategy_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDeployStrategy, foundry_deploy_strategy, FOUNDRY, DEPLOY_STRATEGY, FoundryContextual)

struct _FoundryDeployStrategyClass
{
  FoundryContextualClass parent_class;

  DexFuture *(*supported) (FoundryDeployStrategy  *self);
  DexFuture *(*deploy)    (FoundryDeployStrategy  *self,
                           int                     pty_fd,
                           DexCancellable         *cancellable);
  DexFuture *(*prepare)   (FoundryDeployStrategy  *self,
                           FoundryProcessLauncher *launcher,
                           FoundryBuildPipeline   *pipeline,
                           int                     pty_fd,
                           DexCancellable         *cancellable);

  /*< private >*/
  gpointer _reserved[13];
};

FOUNDRY_AVAILABLE_IN_ALL
DexFuture            *foundry_deploy_strategy_new             (FoundryBuildPipeline   *pipeline);
FOUNDRY_AVAILABLE_IN_ALL
PeasPluginInfo       *foundry_deploy_strategy_dup_plugin_info (FoundryDeployStrategy  *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryBuildPipeline *foundry_deploy_strategy_dup_pipeline    (FoundryDeployStrategy  *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture            *foundry_deploy_strategy_deploy          (FoundryDeployStrategy  *self,
                                                               int                     pty_fd,
                                                               DexCancellable         *cancellable);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture            *foundry_deploy_strategy_supported       (FoundryDeployStrategy  *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture            *foundry_deploy_strategy_prepare         (FoundryDeployStrategy  *self,
                                                               FoundryProcessLauncher *launcher,
                                                               FoundryBuildPipeline   *pipeline,
                                                               int                     pty_fd,
                                                               DexCancellable         *cancellable);

G_END_DECLS
