/* plugin-deviced-deploy-strategy.c
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

#include "plugin-deviced-deploy-strategy.h"
#include "plugin-deviced-device.h"
#include "plugin-deviced-dex.h"

#include "../flatpak/plugin-flatpak-bundle-stage.h"
#include "../flatpak/plugin-flatpak-config.h"

struct _PluginDevicedDeployStrategy
{
  FoundryDeployStrategy parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginDevicedDeployStrategy, plugin_deviced_deploy_strategy, FOUNDRY_TYPE_DEPLOY_STRATEGY)

static DexFuture *
plugin_deviced_deploy_strategy_supported (FoundryDeployStrategy *deploy_strategy)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryDevice) device = NULL;
  g_autoptr(FoundryConfig) config = NULL;

  g_assert (PLUGIN_IS_DEVICED_DEPLOY_STRATEGY (deploy_strategy));

  pipeline = foundry_deploy_strategy_dup_pipeline (deploy_strategy);
  device = foundry_build_pipeline_dup_device (pipeline);
  config = foundry_build_pipeline_dup_config (pipeline);

  if (!PLUGIN_IS_DEVICED_DEVICE (device) || !PLUGIN_IS_FLATPAK_CONFIG (config))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_SUPPORTED,
                                  "Not supported");

  return dex_future_new_for_int (1000);
}

static void
progress_cb (goffset  current,
             goffset  total,
             gpointer user_data)
{
  /* TODO: */
}

static DexFuture *
plugin_deviced_deploy_strategy_deploy_fiber (gpointer data)
{
  FoundryPair *pair = data;
  PluginDevicedDeployStrategy *self = PLUGIN_DEVICED_DEPLOY_STRATEGY (pair->first);
  FoundryBuildProgress *progress = FOUNDRY_BUILD_PROGRESS (pair->second);
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryDevice) device = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) bundle = NULL;
  g_autofree char *app_id = NULL;
  guint n_stages;

  g_assert (pair != NULL);
  g_assert (PLUGIN_IS_DEVICED_DEPLOY_STRATEGY (self));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));

  pipeline = foundry_deploy_strategy_dup_pipeline (FOUNDRY_DEPLOY_STRATEGY (self));
  config = foundry_build_pipeline_dup_config (pipeline);
  device = foundry_build_pipeline_dup_device (pipeline);

  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (PLUGIN_IS_DEVICED_DEVICE (device));

  n_stages = g_list_model_get_n_items (G_LIST_MODEL (pipeline));

  for (guint i = 0; i < n_stages; i++)
    {
      g_autoptr(FoundryBuildStage) stage = g_list_model_get_item (G_LIST_MODEL (pipeline), i);

      if (PLUGIN_IS_FLATPAK_BUNDLE_STAGE (stage))
        {
          bundle = plugin_flatpak_bundle_stage_dup_bundle (PLUGIN_FLATPAK_BUNDLE_STAGE (stage));
          break;
        }
    }

  if (!dex_await (foundry_build_progress_await (progress), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  app_id = plugin_flatpak_config_dup_id (PLUGIN_FLATPAK_CONFIG (config));

  if (!dex_await (plugin_deviced_device_install_bundle (PLUGIN_DEVICED_DEVICE (device),
                                                        g_file_peek_path (bundle),
                                                        progress_cb,
                                                        g_object_ref (progress),
                                                        g_object_unref),
                  &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static DexFuture *
plugin_deviced_deploy_strategy_deploy (FoundryDeployStrategy *deploy_strategy,
                                       int                    pty_fd,
                                       DexCancellable        *cancellable)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(FoundryDevice) device = NULL;
  g_autoptr(FoundryConfig) config = NULL;

  g_assert (PLUGIN_IS_DEVICED_DEPLOY_STRATEGY (deploy_strategy));
  g_assert (pty_fd >= -1);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  pipeline = foundry_deploy_strategy_dup_pipeline (deploy_strategy);
  device = foundry_build_pipeline_dup_device (pipeline);
  config = foundry_build_pipeline_dup_config (pipeline);

  dex_return_error_if_fail (PLUGIN_IS_DEVICED_DEVICE (device));
  dex_return_error_if_fail (PLUGIN_IS_FLATPAK_CONFIG (config));

  progress = foundry_build_pipeline_build (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_EXPORT,
                                           pty_fd,
                                           cancellable);

  return dex_scheduler_spawn (NULL, 0,
                              plugin_deviced_deploy_strategy_deploy_fiber,
                              foundry_pair_new (deploy_strategy, progress),
                              (GDestroyNotify)foundry_pair_free);
}

static gboolean
plugin_deviced_deploy_strategy_prepare_cb (FoundryProcessLauncher  *launcher,
                                           const char * const      *argv,
                                           const char * const      *env,
                                           const char              *cwd,
                                           FoundryUnixFDMap        *unix_fd_map,
                                           gpointer                 user_data,
                                           GError                 **error)
{
  FoundryBuildPipeline *pipeline = user_data;
  g_autoptr(FoundryConfig) config = NULL;
  g_autoptr(FoundryDevice) device = NULL;
  g_autofree char *address_string = NULL;
  g_autofree char *app_id = NULL;
  g_auto(GStrv) environ = NULL;
  guint port;
  guint length;
  int pty_fd = -1;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));

  if (!foundry_unix_fd_map_stdin_isatty (unix_fd_map))
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_FAILED,
                   "Cannot spawn application with additional tooling on remote host");
      return FALSE;
    }

  if ((length = foundry_unix_fd_map_get_length (unix_fd_map)))
    {
      for (guint i = 0; i < length; i++)
        {
          int source_fd;
          int dest_fd;

          source_fd = foundry_unix_fd_map_peek (unix_fd_map, i, &dest_fd);

          if (source_fd == -1 || dest_fd == -1)
            continue;

          if (dest_fd > STDERR_FILENO)
            {
              g_set_error (error,
                           G_IO_ERROR,
                           G_IO_ERROR_FAILED,
                           "Cannot connect file-descriptor (%d:%d) to remote process",
                           source_fd, dest_fd);
              return FALSE;
            }

          if (!isatty (source_fd))
            {
              g_set_error (error,
                           G_IO_ERROR,
                           G_IO_ERROR_FAILED,
                           "Only PTY can be connected to remove device (%d:%d)",
                           source_fd, dest_fd);
              return FALSE;
            }

          if (pty_fd == -1)
            pty_fd = dest_fd;
        }
    }

  if (pty_fd == -1)
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_FAILED,
                   "No PTY provided for application to use");
      return FALSE;
    }

  if (!foundry_process_launcher_merge_unix_fd_map (launcher, unix_fd_map, error))
    return FALSE;

  config = foundry_build_pipeline_dup_config (pipeline);
  device = foundry_build_pipeline_dup_device (pipeline);
  app_id = plugin_flatpak_config_dup_id (PLUGIN_FLATPAK_CONFIG (config));

  if (!(address_string = plugin_deviced_device_dup_network_address (PLUGIN_DEVICED_DEVICE (device), &port, error)))
    return FALSE;

  environ = g_get_environ ();

  foundry_process_launcher_set_environ (launcher, (const char * const *)environ);
  foundry_process_launcher_append_argv (launcher, LIBEXECDIR"/foundry-deviced");
  foundry_process_launcher_append_argv (launcher, "--timeout=10");
  foundry_process_launcher_append_formatted (launcher, "--app-id=%s", app_id);
  foundry_process_launcher_append_formatted (launcher, "--port=%u", port);
  foundry_process_launcher_append_formatted (launcher, "--pty-fd=%d", pty_fd);
  foundry_process_launcher_append_formatted (launcher, "--address=%s", address_string);

  /* TODO: We could possibly connect args to --command= with
   * flatpak and allow proxying FDs between hosts, although
   * that is probably better to implement using Bonsai instead.
   * We would have to teach deviced to connect multiple FDs
   * anyway so things like gdb work w/ stdin/out + pty on fd 3.
   */

  return TRUE;
}

static DexFuture *
plugin_deviced_deploy_strategy_prepare (FoundryDeployStrategy  *deploy_strategy,
                                        FoundryProcessLauncher *launcher,
                                        FoundryBuildPipeline   *pipeline,
                                        int                     pty_fd,
                                        DexCancellable         *cancellable)
{
  g_assert (PLUGIN_IS_DEVICED_DEPLOY_STRATEGY (deploy_strategy));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (pty_fd >= -1);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  foundry_process_launcher_push (launcher,
                                 plugin_deviced_deploy_strategy_prepare_cb,
                                 g_object_ref (pipeline),
                                 g_object_unref);

  foundry_process_launcher_take_fd (launcher, dup (pty_fd), STDIN_FILENO);
  foundry_process_launcher_take_fd (launcher, dup (pty_fd), STDOUT_FILENO);
  foundry_process_launcher_take_fd (launcher, dup (pty_fd), STDERR_FILENO);

  return dex_future_new_true ();
}

static void
plugin_deviced_deploy_strategy_class_init (PluginDevicedDeployStrategyClass *klass)
{
  FoundryDeployStrategyClass *deploy_strategy_class = FOUNDRY_DEPLOY_STRATEGY_CLASS (klass);

  deploy_strategy_class->supported = plugin_deviced_deploy_strategy_supported;
  deploy_strategy_class->deploy = plugin_deviced_deploy_strategy_deploy;
  deploy_strategy_class->prepare = plugin_deviced_deploy_strategy_prepare;
}

static void
plugin_deviced_deploy_strategy_init (PluginDevicedDeployStrategy *self)
{
}
