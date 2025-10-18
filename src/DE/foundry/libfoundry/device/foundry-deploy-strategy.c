/* foundry-deploy-strategy.c
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

#include "foundry-build-pipeline.h"
#include "foundry-debug.h"
#include "foundry-deploy-strategy.h"
#include "foundry-process-launcher.h"

typedef struct
{
  PeasPluginInfo       *plugin_info;
  FoundryBuildPipeline *pipeline;
} FoundryDeployStrategyPrivate;

enum {
  PROP_0,
  PROP_PIPELINE,
  PROP_PLUGIN_INFO,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryDeployStrategy, foundry_deploy_strategy, FOUNDRY_TYPE_CONTEXTUAL)

static GParamSpec *properties[N_PROPS];

static void
foundry_deploy_strategy_finalize (GObject *object)
{
  FoundryDeployStrategy *self = (FoundryDeployStrategy *)object;
  FoundryDeployStrategyPrivate *priv = foundry_deploy_strategy_get_instance_private (self);

  g_clear_object (&priv->plugin_info);
  g_clear_object (&priv->pipeline);

  G_OBJECT_CLASS (foundry_deploy_strategy_parent_class)->finalize (object);
}

static void
foundry_deploy_strategy_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  FoundryDeployStrategy *self = FOUNDRY_DEPLOY_STRATEGY (object);

  switch (prop_id)
    {
    case PROP_PIPELINE:
      g_value_take_object (value, foundry_deploy_strategy_dup_pipeline (self));
      break;

    case PROP_PLUGIN_INFO:
      g_value_take_object (value, foundry_deploy_strategy_dup_plugin_info (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_deploy_strategy_set_property (GObject      *object,
                                      guint         prop_id,
                                      const GValue *value,
                                      GParamSpec   *pspec)
{
  FoundryDeployStrategy *self = FOUNDRY_DEPLOY_STRATEGY (object);
  FoundryDeployStrategyPrivate *priv = foundry_deploy_strategy_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_PIPELINE:
      priv->pipeline = g_value_dup_object (value);
      break;

    case PROP_PLUGIN_INFO:
      priv->plugin_info = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_deploy_strategy_class_init (FoundryDeployStrategyClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_deploy_strategy_finalize;
  object_class->get_property = foundry_deploy_strategy_get_property;
  object_class->set_property = foundry_deploy_strategy_set_property;

  properties[PROP_PIPELINE] =
    g_param_spec_object ("pipeline", NULL, NULL,
                         FOUNDRY_TYPE_BUILD_PIPELINE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PLUGIN_INFO] =
    g_param_spec_object ("plugin-info", NULL, NULL,
                         PEAS_TYPE_PLUGIN_INFO,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_deploy_strategy_init (FoundryDeployStrategy *self)
{
}

/**
 * foundry_deploy_strategy_supported:
 * @self: a [class@Foundry.DeployStrategy]
 *
 * Checks if the deploy strategy is supported for the current configuration.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves the priority of the
 *   strategy or rejects with error if unsupported.
 */
DexFuture *
foundry_deploy_strategy_supported (FoundryDeployStrategy *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEPLOY_STRATEGY (self));

  return FOUNDRY_DEPLOY_STRATEGY_GET_CLASS (self)->supported (self);
}

/**
 * foundry_deploy_strategy_deploy:
 * @self: a [class@Foundry.DeployStrategy]
 *
 * Checks if the deploy strategy is deploy for the current configuration.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves the priority of the
 *   strategy or rejects with error if undeploy.
 */
DexFuture *
foundry_deploy_strategy_deploy (FoundryDeployStrategy *self,
                                int                    pty_fd,
                                DexCancellable        *cancellable)

{
  dex_return_error_if_fail (FOUNDRY_IS_DEPLOY_STRATEGY (self));
  dex_return_error_if_fail (pty_fd >= -1);
  dex_return_error_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  return FOUNDRY_DEPLOY_STRATEGY_GET_CLASS (self)->deploy (self, pty_fd, cancellable);
}

/**
 * foundry_deploy_strategy_dup_pipeline:
 * @self: a [class@Foundry.DeployStrategy]
 *
 * Returns: (transfer full): a [class@Foundry.BuildPipeline]
 */
FoundryBuildPipeline *
foundry_deploy_strategy_dup_pipeline (FoundryDeployStrategy *self)
{
  FoundryDeployStrategyPrivate *priv = foundry_deploy_strategy_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_DEPLOY_STRATEGY (self), NULL);

  return g_object_ref (priv->pipeline);
}

/**
 * foundry_deploy_strategy_dup_plugin_info:
 * @self: a [class@Foundry.DeployStrategy]
 *
 * Returns: (transfer full): a [class@Peas.PluginInfo]
 */
PeasPluginInfo *
foundry_deploy_strategy_dup_plugin_info (FoundryDeployStrategy *self)
{
  FoundryDeployStrategyPrivate *priv = foundry_deploy_strategy_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_DEPLOY_STRATEGY (self), NULL);

  return g_object_ref (priv->plugin_info);
}

static DexFuture *
foundry_deploy_strategy_new_fiber (gpointer data)
{
  FoundryBuildPipeline *pipeline = data;
  g_autoptr(PeasExtensionSet) addins = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryDeployStrategy) best = NULL;
  PeasEngine *engine;
  guint n_items;
  int best_priority = G_MININT;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (pipeline));
  engine = peas_engine_get_default ();
  addins = peas_extension_set_new (engine,
                                   FOUNDRY_TYPE_DEPLOY_STRATEGY,
                                   "context", context,
                                   "pipeline", pipeline,
                                   NULL);
  n_items = g_list_model_get_n_items (G_LIST_MODEL (addins));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(GError) error = NULL;
      g_autoptr(FoundryDeployStrategy) deploy_strategy = g_list_model_get_item (G_LIST_MODEL (addins), i);
      int priority = dex_await_int (foundry_deploy_strategy_supported (deploy_strategy), &error);

      if (error != NULL)
        continue;

      if (priority > best_priority)
        g_set_object (&best, deploy_strategy);
    }

  if (best == NULL)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "No deploy stragey was found");

  return dex_future_new_take_object (g_steal_pointer (&best));
}

/**
 * foundry_deploy_strategy_new:
 * @pipeline: a [class@Foundry.BuildPipeline]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [class@Foundry.DeployStrategy] or rejects with error.
 */
DexFuture *
foundry_deploy_strategy_new (FoundryBuildPipeline *pipeline)
{
  dex_return_error_if_fail (FOUNDRY_IS_BUILD_PIPELINE (pipeline));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_deploy_strategy_new_fiber,
                              g_object_ref (pipeline),
                              g_object_unref);
}

/**
 * foundry_deploy_strategy_prepare:
 * @self: a [class@Foundry.DeployStrategy]
 * @launcher: a [class@Foundry.ProcessLauncher] to prepare
 * @pipeline: a [class@Foundry.BuildPipeline] containing the build stages
 * @pty_fd: the PTY device to use for stdin/stdout/stderr, or -1
 * @cancellable: (nullable): an optional [class@Dex.Cancellable] for cancellation
 *
 * Prepares @launcher to be able to run a command on the device with access
 * to a deployed installation of @pipeline.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any value.
 */
DexFuture *
foundry_deploy_strategy_prepare (FoundryDeployStrategy  *self,
                                 FoundryProcessLauncher *launcher,
                                 FoundryBuildPipeline   *pipeline,
                                 int                     pty_fd,
                                 DexCancellable         *cancellable)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEPLOY_STRATEGY (self));
  dex_return_error_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));
  dex_return_error_if_fail (FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  dex_return_error_if_fail (pty_fd >= -1);
  dex_return_error_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  return FOUNDRY_DEPLOY_STRATEGY_GET_CLASS (self)->prepare (self, launcher, pipeline, pty_fd, cancellable);
}
