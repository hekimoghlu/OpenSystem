/* foundry-run-manager.c
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

#include <glib/gstdio.h>

#include <libpeas.h>

#include "foundry-build-progress.h"
#include "foundry-build-pipeline.h"
#include "foundry-command.h"
#include "foundry-config.h"
#include "foundry-config-manager.h"
#include "foundry-debug.h"
#include "foundry-deploy-strategy.h"
#include "foundry-no-run-tool-private.h"
#include "foundry-process-launcher.h"
#include "foundry-run-tool-private.h"
#include "foundry-service-private.h"
#include "foundry-run-manager.h"
#include "foundry-run-tool-private.h"
#include "foundry-sdk.h"

struct _FoundryRunManager
{
  FoundryService parent_instance;
  int            default_pty_fd;
  guint          busy : 1;
};

struct _FoundryRunManagerClass
{
  FoundryServiceClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryRunManager, foundry_run_manager, FOUNDRY_TYPE_SERVICE)

typedef FoundryRunManager FoundryRunManagerBusy;

G_GNUC_WARN_UNUSED_RESULT
static FoundryRunManagerBusy *
foundry_run_manager_disable_actions (FoundryRunManager *self)
{
  g_assert (FOUNDRY_IS_RUN_MANAGER (self));

  if (self->busy)
    return NULL;

  self->busy = TRUE;

  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "run", FALSE);

  return self;
}

static void
foundry_run_manager_enable_actions (FoundryRunManagerBusy *self)
{
  g_assert (FOUNDRY_IS_RUN_MANAGER (self));
  g_assert (self->busy == TRUE);

  self->busy = FALSE;

  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "run", TRUE);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (FoundryRunManagerBusy, foundry_run_manager_enable_actions)

static DexFuture *
foundry_run_manager_run_action_fiber (gpointer user_data)
{
  FoundryRunManager *self = user_data;
  g_autoptr(FoundryRunManagerBusy) busy = NULL;
  g_autoptr(FoundryConfigManager) config_manager = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryCommand) command = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autoptr(GError) error = NULL;
  g_autofd int build_pty_fd = -1;
  g_autofd int run_pty_fd = -1;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_RUN_MANAGER (self));

  if (!(busy = foundry_run_manager_disable_actions (self)))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_BUSY,
                                  "Service Busy");

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  build_manager = foundry_context_dup_build_manager (context);
  config_manager = foundry_context_dup_config_manager (context);

  build_pty_fd = dup (foundry_build_manager_get_default_pty (build_manager));
  run_pty_fd = dup (self->default_pty_fd);

  if (!(config = foundry_config_manager_dup_config (config_manager)))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "No active configuration");

  /* TODO: Handle command set explicitely on run manager */

  if (!(command = foundry_config_dup_default_command (config)))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "No default command for configuration");

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  /* TODO: Handle most recent tool usage. */

  if (!dex_await (foundry_run_manager_run_command (self, pipeline, command, NULL, build_pty_fd, run_pty_fd, NULL), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static void
foundry_run_manager_run_action (FoundryService *service,
                                const char     *action_name,
                                GVariant       *param)
{
  g_assert (FOUNDRY_IS_RUN_MANAGER (service));

  dex_future_disown (foundry_run_manager_run (FOUNDRY_RUN_MANAGER (service)));
}

static void
foundry_run_manager_finalize (GObject *object)
{
  FoundryRunManager *self = (FoundryRunManager *)object;

  g_clear_fd (&self->default_pty_fd, NULL);

  G_OBJECT_CLASS (foundry_run_manager_parent_class)->finalize (object);
}

static void
foundry_run_manager_class_init (FoundryRunManagerClass *klass)
{
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_run_manager_finalize;

  foundry_service_class_set_action_prefix (service_class, "run-manager");
  foundry_service_class_install_action (service_class, "run", NULL, foundry_run_manager_run_action);
}

static void
foundry_run_manager_init (FoundryRunManager *self)
{
}

/**
 * foundry_run_manager_list_tools:
 * @self: a #FoundryRunManager
 *
 * Gets the available tools that can be used to run the program.
 *
 * Returns: (transfer full): a list of tools supported by the run manager
 *   such as "gdb" or "valgrind" or "sysprof".
 */
char **
foundry_run_manager_list_tools (FoundryRunManager *self)
{
  g_autoptr(PeasExtensionSet) addins = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GStrvBuilder) builder = NULL;
  guint n_items;

  g_return_val_if_fail (FOUNDRY_IS_RUN_MANAGER (self), NULL);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  addins = peas_extension_set_new (peas_engine_get_default (),
                                   FOUNDRY_TYPE_RUN_TOOL,
                                   "context", context,
                                   NULL);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (addins));
  builder = g_strv_builder_new ();

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryRunTool) run_tool = g_list_model_get_item (G_LIST_MODEL (addins), i);
      g_autoptr(PeasPluginInfo) plugin_info = NULL;

      if ((plugin_info = foundry_run_tool_dup_plugin_info (run_tool)))
        g_strv_builder_add (builder, peas_plugin_info_get_module_name (plugin_info));
    }

  return g_strv_builder_end (builder);
}

typedef struct _Run
{
  FoundryRunTool         *run_tool;
  FoundryBuildPipeline   *pipeline;
  FoundryCommand         *command;
  FoundryProcessLauncher *launcher;
  DexCancellable         *cancellable;
  int                     build_pty_fd;
  int                     run_pty_fd;
} Run;

static void
run_free (Run *state)
{
  g_clear_object (&state->run_tool);
  g_clear_object (&state->pipeline);
  g_clear_object (&state->command);
  g_clear_object (&state->launcher);
  g_clear_fd (&state->build_pty_fd, NULL);
  g_clear_fd (&state->run_pty_fd, NULL);
  dex_clear (&state->cancellable);
  g_free (state);
}

static DexFuture *
foundry_run_manager_run_fiber (gpointer data)
{
  Run *state = data;
  g_autoptr(FoundryDeployStrategy) deploy_strategy = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(FoundrySdk) sdk = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_RUN_TOOL (state->run_tool));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (state->pipeline));
  g_assert (FOUNDRY_IS_COMMAND (state->command));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (state->launcher));
  g_assert (!state->cancellable || DEX_IS_CANCELLABLE (state->cancellable));
  g_assert (state->build_pty_fd >= -1);
  g_assert (state->run_pty_fd >= -1);

  sdk = foundry_build_pipeline_dup_sdk (state->pipeline);

  if (!(deploy_strategy = dex_await_object (foundry_deploy_strategy_new (state->pipeline), &error)) ||
      !dex_await (foundry_deploy_strategy_deploy (deploy_strategy, state->build_pty_fd, state->cancellable), &error) ||
      !dex_await (foundry_deploy_strategy_prepare (deploy_strategy, state->launcher, state->pipeline, state->build_pty_fd, state->cancellable), &error) ||
      !dex_await (foundry_run_tool_prepare (state->run_tool, state->pipeline, state->command, state->launcher, state->run_pty_fd), &error) ||
      !(subprocess = foundry_process_launcher_spawn (state->launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  foundry_run_tool_set_subprocess (state->run_tool, subprocess);

  return dex_future_new_take_object (g_object_ref (state->run_tool));
}

/**
 * foundry_run_manager_run_command:
 * @self: a [class@Foundry.RunManager]
 * @pipeline: a [class@Foundry.BuildPipeline]
 * @command: a [class@Foundry.Command]
 *
 * Starts running a program.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [class@Foundry.RunTool].
 */
DexFuture *
foundry_run_manager_run_command (FoundryRunManager    *self,
                                 FoundryBuildPipeline *pipeline,
                                 FoundryCommand       *command,
                                 const char           *tool,
                                 int                   build_pty_fd,
                                 int                   run_pty_fd,
                                 DexCancellable       *cancellable)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GObject) object = NULL;
  PeasPluginInfo *plugin_info;
  PeasEngine *engine;
  Run *state;

  dex_return_error_if_fail (FOUNDRY_IS_RUN_MANAGER (self));
  dex_return_error_if_fail (FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  dex_return_error_if_fail (FOUNDRY_IS_COMMAND (command));
  dex_return_error_if_fail (build_pty_fd >= -1);
  dex_return_error_if_fail (run_pty_fd >= -1);
  dex_return_error_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  engine = peas_engine_get_default ();

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  if (context == NULL)
    goto reject;

  if (tool == NULL || foundry_str_equal0 (tool, "no"))
    {
      object = g_object_new (FOUNDRY_TYPE_NO_RUN_TOOL,
                             "context", context,
                             NULL);
    }
  else
    {
      plugin_info = peas_engine_get_plugin_info (engine, tool);
      if (plugin_info == NULL)
        goto reject;

      object = peas_engine_create_extension (engine,
                                             plugin_info,
                                             FOUNDRY_TYPE_RUN_TOOL,
                                             "context", context,
                                             NULL);
      if (object == NULL)
        goto reject;
    }

  state = g_new0 (Run, 1);
  state->command = g_object_ref (command);
  state->pipeline = g_object_ref (pipeline);
  state->run_tool = g_object_ref (FOUNDRY_RUN_TOOL (object));
  state->launcher = foundry_process_launcher_new ();
  state->build_pty_fd = build_pty_fd >= 0 ? dup (build_pty_fd) : -1;
  state->run_pty_fd = run_pty_fd >= 0 ? dup (run_pty_fd) : -1;
  state->cancellable = cancellable ? dex_ref (cancellable) : dex_cancellable_new ();

  return dex_scheduler_spawn (NULL, 0,
                              foundry_run_manager_run_fiber,
                              state,
                              (GDestroyNotify) run_free);

reject:
  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Cannot find tool \"%s\"",
                                tool);
}

void
foundry_run_manager_set_default_pty (FoundryRunManager *self,
                                     int                pty_fd)
{
  g_return_if_fail (FOUNDRY_IS_RUN_MANAGER (self));

  g_clear_fd (&self->default_pty_fd, NULL);

  if (pty_fd > -1)
    self->default_pty_fd = dup (pty_fd);
}

/**
 * foundry_run_manager_run:
 * @self: a [class@Foundry.RunManager]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any value
 *   or rejects with error.
 */
DexFuture *
foundry_run_manager_run (FoundryRunManager *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_RUN_MANAGER (self));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_run_manager_run_action_fiber,
                              g_object_ref (self),
                              g_object_unref);
}
