/* plugin-local-device-deploy-strategy.c
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

#include "plugin-local-device-deploy-strategy.h"

struct _PluginLocalDeviceDeployStrategy
{
  FoundryDeployStrategy parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginLocalDeviceDeployStrategy, plugin_local_device_deploy_strategy, FOUNDRY_TYPE_DEPLOY_STRATEGY)

static DexFuture *
plugin_local_device_deploy_strategy_prepare (FoundryDeployStrategy  *deploy_strategy,
                                             FoundryProcessLauncher *launcher,
                                             FoundryBuildPipeline   *pipeline,
                                             int                     pty_fd,
                                             DexCancellable         *cancellable)
{
  g_autoptr(FoundrySdk) sdk = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_LOCAL_DEVICE_DEPLOY_STRATEGY (deploy_strategy));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (pty_fd >= -1);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  /* TODO: use sdk for prepare_to_run */
  /* TODO: setup PATH, LD_LIBRARY_PATH, etc */

  sdk = foundry_build_pipeline_dup_sdk (pipeline);

  return foundry_sdk_prepare_to_run (sdk, pipeline, launcher);
}

static DexFuture *
plugin_local_device_deploy_strategy_supported (FoundryDeployStrategy *deploy_strategy)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryDevice) device = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_LOCAL_DEVICE_DEPLOY_STRATEGY (deploy_strategy));

  pipeline = foundry_deploy_strategy_dup_pipeline (deploy_strategy);
  device = foundry_build_pipeline_dup_device (pipeline);

  if (FOUNDRY_IS_LOCAL_DEVICE (device))
    return dex_future_new_for_int (0);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Device \"%s\" is not supported",
                                G_OBJECT_TYPE_NAME (device));
}

static DexFuture *
plugin_local_device_deploy_strategy_deploy (FoundryDeployStrategy *deploy_strategy,
                                            int                    pty_fd,
                                            DexCancellable        *cancellable)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_LOCAL_DEVICE_DEPLOY_STRATEGY (deploy_strategy));
  g_assert (pty_fd >= -1);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  pipeline = foundry_deploy_strategy_dup_pipeline (deploy_strategy);
  progress = foundry_build_pipeline_build (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_INSTALL,
                                           pty_fd,
                                           cancellable);

  return foundry_build_progress_await (progress);
}

static void
plugin_local_device_deploy_strategy_class_init (PluginLocalDeviceDeployStrategyClass *klass)
{
  FoundryDeployStrategyClass *deploy_strategy_class = FOUNDRY_DEPLOY_STRATEGY_CLASS (klass);

  deploy_strategy_class->deploy = plugin_local_device_deploy_strategy_deploy;
  deploy_strategy_class->prepare = plugin_local_device_deploy_strategy_prepare;
  deploy_strategy_class->supported = plugin_local_device_deploy_strategy_supported;
}

static void
plugin_local_device_deploy_strategy_init (PluginLocalDeviceDeployStrategy *self)
{
}
