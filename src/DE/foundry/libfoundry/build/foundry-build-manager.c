/* foundry-build-manager.c
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

#include "config.h"

#include <glib/gi18n-lib.h>
#include <glib/gstdio.h>

#include "foundry-build-manager.h"
#include "foundry-build-pipeline-private.h"
#include "foundry-build-progress.h"
#include "foundry-config.h"
#include "foundry-config-manager.h"
#include "foundry-debug.h"
#include "foundry-device.h"
#include "foundry-device-manager.h"
#include "foundry-sdk.h"
#include "foundry-sdk-manager.h"
#include "foundry-service-private.h"

struct _FoundryBuildManager
{
  FoundryService  parent_instance;
  DexCancellable *cancellable;
  DexFuture      *pipeline;
  int             default_pty_fd;
  guint           busy : 1;
};

struct _FoundryBuildManagerClass
{
  FoundryServiceClass parent_instance;
};

G_DEFINE_QUARK (foundry_build_error, foundry_build_error)
G_DEFINE_FINAL_TYPE (FoundryBuildManager, foundry_build_manager, FOUNDRY_TYPE_SERVICE)

enum {
  PROP_0,
  PROP_BUSY,
  N_PROPS
};

enum {
  PIPELINE_INVALIDATED,
  N_SIGNALS
};

static GParamSpec *properties[N_PROPS];
static guint signals[N_SIGNALS];

typedef FoundryBuildManager FoundryBuildManagerBusy;

static DexCancellable *
foundry_build_manager_dup_cancellable (FoundryBuildManager *self)
{
  g_assert (FOUNDRY_IS_BUILD_MANAGER (self));

  if (self->cancellable != NULL &&
      dex_future_is_rejected (DEX_FUTURE (self->cancellable)))
    dex_clear (&self->cancellable);

  if (self->cancellable == NULL)
    self->cancellable = dex_cancellable_new ();

  return dex_ref (self->cancellable);
}

static FoundryBuildManagerBusy *
foundry_build_manager_disable_actions (FoundryBuildManager *self)
{
  g_assert (FOUNDRY_IS_BUILD_MANAGER (self));

  if (self->busy)
    return NULL;

  self->busy = TRUE;

  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "build", FALSE);
  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "rebuild", FALSE);
  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "clean", FALSE);
  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "invalidate", FALSE);
  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "purge", FALSE);

  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "stop", TRUE);

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_BUSY]);

  return self;
}

static void
foundry_build_manager_enable_actions (FoundryBuildManager *self)
{
  g_assert (FOUNDRY_IS_BUILD_MANAGER (self));

  self->busy = FALSE;

  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "build", TRUE);
  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "rebuild", TRUE);
  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "clean", TRUE);
  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "invalidate", TRUE);
  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "purge", TRUE);

  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "stop", FALSE);

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_BUSY]);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (FoundryBuildManagerBusy, foundry_build_manager_enable_actions)

static DexFuture *
foundry_build_manager_build_action_fiber (gpointer data)
{
  FoundryBuildManager *self = data;
  g_autoptr(FoundryBuildManagerBusy) busy = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_BUILD_MANAGER (self));

  if (!(busy = foundry_build_manager_disable_actions (self)))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_BUSY,
                                  "Service busy");

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (self), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  cancellable = foundry_build_manager_dup_cancellable (self);

  progress = foundry_build_pipeline_build (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_BUILD,
                                           self->default_pty_fd,
                                           cancellable);

  if (!dex_await (foundry_build_progress_await (progress), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static void
foundry_build_manager_build_action (FoundryService *service,
                                    const char     *action_name,
                                    GVariant       *param)
{
  g_assert (FOUNDRY_IS_BUILD_MANAGER (service));

  dex_future_disown (foundry_build_manager_build (FOUNDRY_BUILD_MANAGER (service)));
}

static DexFuture *
foundry_build_manager_clean_action_fiber (gpointer data)
{
  FoundryBuildManager *self = data;
  g_autoptr(FoundryBuildManagerBusy) busy = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_BUILD_MANAGER (self));

  if (!(busy = foundry_build_manager_disable_actions (self)))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_BUSY,
                                  "Service busy");

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (self), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  cancellable = foundry_build_manager_dup_cancellable (self);

  progress = foundry_build_pipeline_clean (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_BUILD,
                                           self->default_pty_fd,
                                           cancellable);

  if (!dex_await (foundry_build_progress_await (progress), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static void
foundry_build_manager_clean_action (FoundryService *service,
                                    const char     *action_name,
                                    GVariant       *param)
{
  g_assert (FOUNDRY_IS_BUILD_MANAGER (service));

  dex_future_disown (foundry_build_manager_clean (FOUNDRY_BUILD_MANAGER (service)));
}

static DexFuture *
foundry_build_manager_purge_action_fiber (gpointer data)
{
  FoundryBuildManager *self = data;
  g_autoptr(FoundryBuildManagerBusy) busy = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_BUILD_MANAGER (self));

  if (!(busy = foundry_build_manager_disable_actions (self)))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_BUSY,
                                  "Service busy");

  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (self), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  cancellable = foundry_build_manager_dup_cancellable (self);

  progress = foundry_build_pipeline_purge (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_BUILD,
                                           self->default_pty_fd,
                                           cancellable);

  if (!dex_await (foundry_build_progress_await (progress), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static DexFuture *
foundry_build_manager_rebuild_action_fiber (gpointer data)
{
  FoundryBuildManager *self = data;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_BUILD_MANAGER (self));

  if (!dex_await (foundry_build_manager_purge (self), &error) ||
      !dex_await (foundry_build_manager_build (self), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static void
foundry_build_manager_purge_action (FoundryService *service,
                                    const char     *action_name,
                                    GVariant       *param)
{
  g_assert (FOUNDRY_IS_BUILD_MANAGER (service));

  dex_future_disown (foundry_build_manager_purge (FOUNDRY_BUILD_MANAGER (service)));
}

static void
foundry_build_manager_invalidate_action (FoundryService *service,
                                         const char     *action_name,
                                         GVariant       *param)
{
  g_assert (FOUNDRY_IS_BUILD_MANAGER (service));

  foundry_build_manager_invalidate (FOUNDRY_BUILD_MANAGER (service));
}

static void
foundry_build_manager_rebuild_action (FoundryService *service,
                                      const char     *action_name,
                                      GVariant       *param)
{
  g_assert (FOUNDRY_IS_BUILD_MANAGER (service));

  dex_future_disown (foundry_build_manager_rebuild (FOUNDRY_BUILD_MANAGER (service)));
}

static void
foundry_build_manager_stop_action (FoundryService *service,
                                   const char     *action_name,
                                   GVariant       *param)
{
  g_assert (FOUNDRY_IS_BUILD_MANAGER (service));

  foundry_build_manager_stop (FOUNDRY_BUILD_MANAGER (service));
}

static void
foundry_build_manager_constructed (GObject *object)
{
  FoundryBuildManager *self = (FoundryBuildManager *)object;

  G_OBJECT_CLASS (foundry_build_manager_parent_class)->constructed (object);

  foundry_service_action_set_enabled (FOUNDRY_SERVICE (self), "stop", FALSE);
}

static void
foundry_build_manager_finalize (GObject *object)
{
  FoundryBuildManager *self = (FoundryBuildManager *)object;

  g_clear_fd (&self->default_pty_fd, NULL);
  dex_clear (&self->cancellable);

  G_OBJECT_CLASS (foundry_build_manager_parent_class)->finalize (object);
}

static void
foundry_build_manager_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundryBuildManager *self = FOUNDRY_BUILD_MANAGER (object);

  switch (prop_id)
    {
    case PROP_BUSY:
      g_value_set_boolean (value, foundry_build_manager_get_busy (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_build_manager_class_init (FoundryBuildManagerClass *klass)
{
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->constructed = foundry_build_manager_constructed;
  object_class->finalize = foundry_build_manager_finalize;
  object_class->get_property = foundry_build_manager_get_property;

  properties[PROP_BUSY] =
    g_param_spec_boolean ("busy", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  foundry_service_class_set_action_prefix (service_class, "build-manager");
  foundry_service_class_install_action (service_class, "build", NULL, foundry_build_manager_build_action);
  foundry_service_class_install_action (service_class, "clean", NULL, foundry_build_manager_clean_action);
  foundry_service_class_install_action (service_class, "purge", NULL, foundry_build_manager_purge_action);
  foundry_service_class_install_action (service_class, "invalidate", NULL, foundry_build_manager_invalidate_action);
  foundry_service_class_install_action (service_class, "rebuild", NULL, foundry_build_manager_rebuild_action);
  foundry_service_class_install_action (service_class, "stop", NULL, foundry_build_manager_stop_action);

  /**
   * FoundryBuildManager::pipeline-invalidated:
   *
   * This signal is emitted when a loaded pipeline has become invalidated.
   * Observers of pipelines may want to call
   * [method@Foundry.BuildManager.load_pipeline] to request a new pipeline.
   */
  signals[PIPELINE_INVALIDATED] =
    g_signal_new ("pipeline-invalidated",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  NULL,
                  G_TYPE_NONE, 0);
}

static void
foundry_build_manager_init (FoundryBuildManager *self)
{
  self->default_pty_fd = -1;
}

static DexFuture *
foundry_build_manager_load_pipeline_fiber (gpointer user_data)
{
  FoundryBuildManager *self = user_data;
  g_autoptr(FoundryConfigManager) config_manager = NULL;
  g_autoptr(FoundryDeviceManager) device_manager = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundrySdkManager) sdk_manager = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autoptr(FoundryDevice) device = NULL;
  g_autoptr(FoundrySdk) sdk = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_BUILD_MANAGER (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  dex_return_error_if_fail (FOUNDRY_IS_CONTEXT (context));

  if (foundry_context_is_shared (context))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_SUPPORTED,
                                  "Building is not supported in shared mode");

  config_manager = foundry_context_dup_config_manager (context);
  dex_return_error_if_fail (FOUNDRY_IS_CONFIG_MANAGER (config_manager));

  device_manager = foundry_context_dup_device_manager (context);
  dex_return_error_if_fail (FOUNDRY_IS_DEVICE_MANAGER (device_manager));

  sdk_manager = foundry_context_dup_sdk_manager (context);
  dex_return_error_if_fail (FOUNDRY_IS_SDK_MANAGER (sdk_manager));

  if (!dex_await (dex_future_all (foundry_service_when_ready (FOUNDRY_SERVICE (self)),
                                  foundry_service_when_ready (FOUNDRY_SERVICE (config_manager)),
                                  foundry_service_when_ready (FOUNDRY_SERVICE (device_manager)),
                                  foundry_service_when_ready (FOUNDRY_SERVICE (sdk_manager)),
                                  NULL),
                  &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(config = foundry_config_manager_dup_config (config_manager)))
    return dex_future_new_reject (FOUNDRY_BUILD_ERROR,
                                  FOUNDRY_BUILD_ERROR_INVALID_CONFIG,
                                  _("Project does not contain an active build configuration"));

  if (!(device = foundry_device_manager_dup_device (device_manager)))
    return dex_future_new_reject (FOUNDRY_BUILD_ERROR,
                                  FOUNDRY_BUILD_ERROR_INVALID_DEVICE,
                                  _("Project does not contain an active build device"));

  if (!(sdk = foundry_sdk_manager_dup_sdk (sdk_manager)))
    return dex_future_new_reject (FOUNDRY_BUILD_ERROR,
                                  FOUNDRY_BUILD_ERROR_INVALID_SDK,
                                  _("Project does not contain an active SDK"));

  if (!(pipeline = dex_await_object (foundry_build_pipeline_new (context, config, device, sdk, TRUE), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!dex_await (_foundry_build_pipeline_load (pipeline), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_take_object (g_steal_pointer (&pipeline));
}

/**
 * foundry_build_manager_load_pipeline:
 * @self: a #FoundryBuildManager
 *
 * Loads the pipeline as a future.
 *
 * If the pipeline is already being loaded, the future will be completed
 * as part of that request.
 *
 * If the pipeline is already loaded, the future returned will already
 * be resolved.
 *
 * Otherwise, a new request to load the pipeline is created and the future
 * will resolve upon completion.
 *
 * Returns: (transfer full): a #DexFuture that resolves to a #FoundryPipeline
 */
DexFuture *
foundry_build_manager_load_pipeline (FoundryBuildManager *self)
{
  g_return_val_if_fail (FOUNDRY_IS_BUILD_MANAGER (self), NULL);

  if (self->pipeline == NULL)
    self->pipeline = dex_scheduler_spawn (dex_scheduler_get_default (), 0,
                                          foundry_build_manager_load_pipeline_fiber,
                                          g_object_ref (self),
                                          g_object_unref);

  return dex_ref (DEX_FUTURE (self->pipeline));
}

void
foundry_build_manager_invalidate (FoundryBuildManager *self)
{
  g_return_if_fail (FOUNDRY_IS_BUILD_MANAGER (self));

  if (self->pipeline != NULL)
    {
      dex_clear (&self->pipeline);
      g_signal_emit (self, signals[PIPELINE_INVALIDATED], 0);
    }
}

void
foundry_build_manager_set_default_pty (FoundryBuildManager *self,
                                       int                  pty_fd)
{
  g_return_if_fail (FOUNDRY_IS_BUILD_MANAGER (self));

  g_clear_fd (&self->default_pty_fd, NULL);

  if (pty_fd > -1)
    self->default_pty_fd = dup (pty_fd);
}

int
foundry_build_manager_get_default_pty (FoundryBuildManager *self)
{
  g_return_val_if_fail (FOUNDRY_IS_BUILD_MANAGER (self), -1);

  return self->default_pty_fd;
}

/**
 * foundry_build_manager_build:
 * @self: a [class@Foundry.BuildManager]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   to any value or rejects with error.
 */
DexFuture *
foundry_build_manager_build (FoundryBuildManager *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_BUILD_MANAGER (self));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_build_manager_build_action_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

/**
 * foundry_build_manager_clean:
 * @self: a [class@Foundry.BuildManager]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   to any value or rejects with error.
 */
DexFuture *
foundry_build_manager_clean (FoundryBuildManager *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_BUILD_MANAGER (self));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_build_manager_clean_action_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

/**
 * foundry_build_manager_purge:
 * @self: a [class@Foundry.BuildManager]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   to any value or rejects with error.
 */
DexFuture *
foundry_build_manager_purge (FoundryBuildManager *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_BUILD_MANAGER (self));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_build_manager_purge_action_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

/**
 * foundry_build_manager_rebuild:
 * @self: a [class@Foundry.BuildManager]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   to any value or rejects with error.
 */
DexFuture *
foundry_build_manager_rebuild (FoundryBuildManager *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_BUILD_MANAGER (self));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_build_manager_rebuild_action_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

/**
 * foundry_build_manager_stop:
 * @self: a [class@Foundry.BuildManager]
 *
 * Stop any active builds controlled by the build manager.
 */
void
foundry_build_manager_stop (FoundryBuildManager *self)
{
  g_return_if_fail (FOUNDRY_IS_BUILD_MANAGER (self));

  if (self->cancellable != NULL)
    dex_cancellable_cancel (self->cancellable);
}

/**
 * foundry_build_manager_get_busy:
 * @self: a [class@Foundry.BuildManager]
 *
 * If the build manager is currently busy running an operation on the
 * active pipeline.
 */
gboolean
foundry_build_manager_get_busy (FoundryBuildManager *self)
{
  g_return_val_if_fail (FOUNDRY_IS_BUILD_MANAGER (self), FALSE);

  return self->busy;
}
