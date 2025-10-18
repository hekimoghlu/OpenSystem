/* foundry-sdk.c
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

#include "foundry-build-pipeline.h"
#include "foundry-config.h"
#include "foundry-contextual-private.h"
#include "foundry-operation.h"
#include "foundry-sdk-manager.h"
#include "foundry-sdk-private.h"
#include "foundry-sdk-provider.h"
#include "foundry-shell-private.h"
#include "foundry-subprocess.h"
#include "foundry-util.h"

typedef struct _FoundrySdkPrivate
{
  GWeakRef provider_wr;
  char *id;
  char *arch;
  char *name;
  char *kind;
  guint extension_only : 1;
  guint installed : 1;
} FoundrySdkPrivate;

enum {
  PROP_0,
  PROP_ACTIVE,
  PROP_ARCH,
  PROP_EXTENSION_ONLY,
  PROP_ID,
  PROP_INSTALLED,
  PROP_KIND,
  PROP_NAME,
  PROP_PROVIDER,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundrySdk, foundry_sdk, FOUNDRY_TYPE_CONTEXTUAL)

static GParamSpec *properties[N_PROPS];

typedef struct _ContainsProgram
{
  FoundryProcessLauncher *launcher;
  FoundrySdk             *self;
  char                   *program;
} ContainsProgram;

static void
contains_program_free (ContainsProgram *state)
{
  g_clear_pointer (&state->program, g_free);
  g_clear_object (&state->self);
  g_clear_object (&state->launcher);
  g_free (state);
}

static DexFuture *
strip_string (DexFuture *completed,
              gpointer   user_data)
{
  g_autofree char *str = dex_await_string (dex_ref (completed), NULL);

  if (str != NULL)
    g_strstrip (str);

  return dex_future_new_take_string (g_steal_pointer (&str));
}

static DexFuture *
foundry_sdk_real_contains_program_cb (DexFuture *completed,
                                      gpointer   user_data)
{
  ContainsProgram *state = user_data;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (state->launcher));
  g_assert (FOUNDRY_IS_SDK (state->self));
  g_assert (state->program != NULL);

  foundry_process_launcher_push_shell (state->launcher, FOUNDRY_PROCESS_LAUNCHER_SHELL_DEFAULT);

  foundry_process_launcher_append_argv (state->launcher, "which");
  foundry_process_launcher_append_argv (state->launcher, state->program);

  if (!(subprocess = foundry_process_launcher_spawn_with_flags (state->launcher,
                                                                G_SUBPROCESS_FLAGS_STDOUT_PIPE,
                                                                &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_then (foundry_subprocess_communicate_utf8 (subprocess, NULL),
                          strip_string, NULL, NULL);
}

static DexFuture *
foundry_sdk_real_contains_program (FoundrySdk *self,
                                   const char *program)
{
  ContainsProgram *state;
  DexFuture *future;

  g_assert (FOUNDRY_IS_SDK (self));
  g_assert (program != NULL);

  state = g_new0 (ContainsProgram, 1);
  state->self = g_object_ref (self);
  state->program = g_strdup (program);
  state->launcher = foundry_process_launcher_new ();

  future = foundry_sdk_prepare_to_build (self, NULL, state->launcher, FOUNDRY_BUILD_PIPELINE_PHASE_BUILD);
  future = dex_future_then (future,
                            foundry_sdk_real_contains_program_cb,
                            state,
                            (GDestroyNotify) contains_program_free);

  return future;
}

static void
foundry_sdk_finalize (GObject *object)
{
  FoundrySdk *self = (FoundrySdk *)object;
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_weak_ref_clear (&priv->provider_wr);

  g_clear_pointer (&priv->id, g_free);
  g_clear_pointer (&priv->name, g_free);
  g_clear_pointer (&priv->kind, g_free);
  g_clear_pointer (&priv->arch, g_free);

  G_OBJECT_CLASS (foundry_sdk_parent_class)->finalize (object);
}

static void
foundry_sdk_get_property (GObject    *object,
                          guint       prop_id,
                          GValue     *value,
                          GParamSpec *pspec)
{
  FoundrySdk *self = FOUNDRY_SDK (object);

  switch (prop_id)
    {
    case PROP_ACTIVE:
      g_value_set_boolean (value, foundry_sdk_get_active (self));
      break;

    case PROP_ARCH:
      g_value_take_string (value, foundry_sdk_dup_arch (self));
      break;

    case PROP_EXTENSION_ONLY:
      g_value_set_boolean (value, foundry_sdk_get_extension_only (self));
      break;

    case PROP_ID:
      g_value_take_string (value, foundry_sdk_dup_id (self));
      break;

    case PROP_INSTALLED:
      g_value_set_boolean (value, foundry_sdk_get_installed (self));
      break;

    case PROP_KIND:
      g_value_take_string (value, foundry_sdk_dup_kind (self));
      break;

    case PROP_NAME:
      g_value_take_string (value, foundry_sdk_dup_name (self));
      break;

    case PROP_PROVIDER:
      g_value_take_object (value, _foundry_sdk_dup_provider (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_sdk_set_property (GObject      *object,
                          guint         prop_id,
                          const GValue *value,
                          GParamSpec   *pspec)
{
  FoundrySdk *self = FOUNDRY_SDK (object);

  switch (prop_id)
    {
    case PROP_ARCH:
      foundry_sdk_set_arch (self, g_value_get_string (value));
      break;

    case PROP_EXTENSION_ONLY:
      foundry_sdk_set_extension_only (self, g_value_get_boolean (value));
      break;

    case PROP_ID:
      foundry_sdk_set_id (self, g_value_get_string (value));
      break;

    case PROP_INSTALLED:
      foundry_sdk_set_installed (self, g_value_get_boolean (value));
      break;

    case PROP_KIND:
      foundry_sdk_set_kind (self, g_value_get_string (value));
      break;

    case PROP_NAME:
      foundry_sdk_set_name (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_sdk_class_init (FoundrySdkClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_sdk_finalize;
  object_class->get_property = foundry_sdk_get_property;
  object_class->set_property = foundry_sdk_set_property;

  klass->contains_program = foundry_sdk_real_contains_program;

  properties[PROP_ACTIVE] =
    g_param_spec_boolean ("active", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_ARCH] =
    g_param_spec_string ("arch", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_EXTENSION_ONLY] =
    g_param_spec_boolean ("extension-only", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_INSTALLED] =
    g_param_spec_boolean ("installed", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_KIND] =
    g_param_spec_string ("kind", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PROVIDER] =
    g_param_spec_object ("provider", NULL, NULL,
                         FOUNDRY_TYPE_SDK_PROVIDER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_sdk_init (FoundrySdk *self)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_weak_ref_init (&priv->provider_wr, NULL);
}

/**
 * foundry_sdk_dup_id:
 * @self: a #FoundrySdk
 *
 * Gets the user-visible id for the SDK.
 *
 * Returns: (transfer full): a newly allocated string
 */
char *
foundry_sdk_dup_id (FoundrySdk *self)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_SDK (self), NULL);

  return g_strdup (priv->id);
}

/**
 * foundry_sdk_set_id:
 * @self: a #FoundrySdk
 *
 * Set the unique id of the SDK.
 *
 * This should only be called by implementations of #FoundrySdkProvider.
 */
void
foundry_sdk_set_id (FoundrySdk *self,
                    const char *id)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_SDK (self));

  if (g_set_str (&priv->id, id))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ID]);
}

/**
 * foundry_sdk_dup_arch:
 * @self: a #FoundrySdk
 *
 * Gets the architecture of the SDK.
 *
 * Returns: (transfer full): a newly allocated string
 */
char *
foundry_sdk_dup_arch (FoundrySdk *self)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_SDK (self), NULL);

  return g_strdup (priv->arch);
}

/**
 * foundry_sdk_set_arch:
 * @self: a #FoundrySdk
 *
 * Set the architecture of the SDK.
 *
 * This should only be called by [class@Foundry.SdkProvider] classes.
 */
void
foundry_sdk_set_arch (FoundrySdk *self,
                      const char *arch)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_SDK (self));

  if (g_set_str (&priv->arch, arch))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ARCH]);
}

/**
 * foundry_sdk_dup_kind:
 * @self: a #FoundrySdk
 *
 * Gets the user-visible kind for the SDK.
 *
 * Returns: (transfer full): a newly allocated string
 */
char *
foundry_sdk_dup_kind (FoundrySdk *self)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_SDK (self), NULL);

  return g_strdup (priv->kind);
}

/**
 * foundry_sdk_set_kind:
 * @self: a #FoundrySdk
 *
 * Set the user-visible kind of the sdk.
 *
 * This should only be called by implementations of #FoundrySdkProvider.
 */
void
foundry_sdk_set_kind (FoundrySdk *self,
                      const char *kind)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_SDK (self));

  if (g_set_str (&priv->kind, kind))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_KIND]);
}

/**
 * foundry_sdk_dup_name:
 * @self: a #FoundrySdk
 *
 * Gets the user-visible name for the SDK.
 *
 * Returns: (transfer full): a newly allocated string
 */
char *
foundry_sdk_dup_name (FoundrySdk *self)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_SDK (self), NULL);

  return g_strdup (priv->name);
}

/**
 * foundry_sdk_set_name:
 * @self: a #FoundrySdk
 *
 * Set the user-visible name of the sdk.
 *
 * This should only be called by implementations of #FoundrySdkProvider.
 */
void
foundry_sdk_set_name (FoundrySdk *self,
                      const char *name)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_SDK (self));

  if (g_set_str (&priv->name, name))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_NAME]);
}

FoundrySdkProvider *
_foundry_sdk_dup_provider (FoundrySdk *self)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_SDK (self), NULL);

  return g_weak_ref_get (&priv->provider_wr);
}

void
_foundry_sdk_set_provider (FoundrySdk         *self,
                           FoundrySdkProvider *provider)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_SDK (self));
  g_return_if_fail (!provider || FOUNDRY_IS_SDK_PROVIDER (provider));

  g_weak_ref_set (&priv->provider_wr, provider);
}

gboolean
foundry_sdk_get_active (FoundrySdk *self)
{
  g_autoptr(FoundryContext) context = NULL;

  g_return_val_if_fail (FOUNDRY_IS_SDK (self), FALSE);

  if ((context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    {
      g_autoptr(FoundrySdkManager) sdk_manager = foundry_context_dup_sdk_manager (context);
      g_autoptr(FoundrySdk) sdk = foundry_sdk_manager_dup_sdk (sdk_manager);

      return sdk == self;
    }

  return FALSE;
}

gboolean
foundry_sdk_get_extension_only (FoundrySdk *self)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_SDK (self), FALSE);

  return priv->extension_only;
}

void
foundry_sdk_set_extension_only (FoundrySdk *self,
                                gboolean    extension_only)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_SDK (self));

  extension_only = !!extension_only;

  if (priv->extension_only != extension_only)
    {
      priv->extension_only = extension_only;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_INSTALLED]);
    }
}

gboolean
foundry_sdk_get_installed (FoundrySdk *self)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_SDK (self), FALSE);

  return priv->installed;
}

void
foundry_sdk_set_installed (FoundrySdk *self,
                           gboolean    installed)
{
  FoundrySdkPrivate *priv = foundry_sdk_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_SDK (self));

  installed = !!installed;

  if (priv->installed != installed)
    {
      priv->installed = installed;

      if (foundry_sdk_get_active (self))
        _foundry_contextual_invalidate_pipeline (FOUNDRY_CONTEXTUAL (self));

      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_INSTALLED]);
    }
}

/**
 * foundry_sdk_prepare_to_build:
 * @self: a #FoundrySdk
 * @pipeline: (nullable): a [class@Foundry.BuildPipeline] or %NULL
 * @launcher: the launcher to prepare
 * @phase: the phase of the build
 *
 * Prepares @launcher to be able to build applications.
 *
 * That may mean setting things up to access a SDK tooling or other compoonents.
 *
 * @phase is used to differentiate between different types of launchers. For
 * example, you may apply a different environment for building dependencies
 * than for the project itself.
 *
 * Returns: (transfer full): a [class@Dex.Future]
 */
DexFuture *
foundry_sdk_prepare_to_build (FoundrySdk                *self,
                              FoundryBuildPipeline      *pipeline,
                              FoundryProcessLauncher    *launcher,
                              FoundryBuildPipelinePhase  phase)
{
  dex_return_error_if_fail (FOUNDRY_IS_SDK (self));
  dex_return_error_if_fail (foundry_sdk_get_installed (self));
  dex_return_error_if_fail (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  dex_return_error_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  if (!foundry_sdk_get_installed (self))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "SDK is not installed");

  if (FOUNDRY_SDK_GET_CLASS (self)->prepare_to_build)
    return FOUNDRY_SDK_GET_CLASS (self)->prepare_to_build (self, pipeline, launcher, phase);

  return dex_future_new_true ();
}

/**
 * foundry_sdk_prepare_to_run:
 * @self: a #FoundrySdk
 * @pipeline: (nullable): a [class@Foundry.BuildPipeline] or %NULL
 * @launcher: the launcher to prepare
 *
 * Prepares @launcher to be able to run applications.
 *
 * That may mean setting things up to access a display server, network,
 * or other compoonents.
 *
 * Returns: (transfer full): a [class@Dex.Future]
 */
DexFuture *
foundry_sdk_prepare_to_run (FoundrySdk             *self,
                            FoundryBuildPipeline   *pipeline,
                            FoundryProcessLauncher *launcher)
{
  dex_return_error_if_fail (FOUNDRY_IS_SDK (self));
  dex_return_error_if_fail (foundry_sdk_get_installed (self));
  dex_return_error_if_fail (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  dex_return_error_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  if (!foundry_sdk_get_installed (self))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "SDK is not installed");

  if (FOUNDRY_SDK_GET_CLASS (self)->prepare_to_run)
    return FOUNDRY_SDK_GET_CLASS (self)->prepare_to_run (self, pipeline, launcher);

  return dex_future_new_true ();
}

/**
 * foundry_sdk_contains_program:
 * @self: a #FoundrySdk
 * @program: the program such as "ps"
 *
 * Looks for @program within the SDK.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a string
 *   containing the path of @program.
 */
DexFuture *
foundry_sdk_contains_program (FoundrySdk *self,
                              const char *program)
{
  dex_return_error_if_fail (FOUNDRY_IS_SDK (self));
  dex_return_error_if_fail (foundry_sdk_get_installed (self));
  dex_return_error_if_fail (program != NULL);

  if (!foundry_sdk_get_installed (self))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "The SDK is not installed");

  return FOUNDRY_SDK_GET_CLASS (self)->contains_program (self, program);
}

static DexFuture *
foundry_sdk_discover_shell_fiber (gpointer user_data)
{
  FoundrySdk *self = user_data;
  const char *default_shell;
  g_autofree char *shell = NULL;

  g_assert (FOUNDRY_IS_SDK (self));

  /* Ensure the shell subsystem has completed startup */
  dex_await (_foundry_shell_init (), NULL);

  /* Now look at what we discovered as the user default */
  default_shell = foundry_shell_get_default ();

  if (default_shell != NULL)
    default_shell = shell = g_path_get_basename (default_shell);

  /* If this is in the SDK, use that */
  if (default_shell != NULL &&
      dex_await (foundry_sdk_contains_program (self, default_shell), NULL))
    return dex_future_new_take_string (g_strdup (default_shell));

  /* If we have bash, fallback to that */
  if (dex_await (foundry_sdk_contains_program (self, "bash"), NULL))
    return dex_future_new_take_string (g_strdup ("bash"));

  /* Okay, just try sh */
  return dex_future_new_take_string (g_strdup ("sh"));
}

/**
 * foundry_sdk_discover_shell:
 * @self: a #FoundrySdk
 *
 * Tries to discover the shell to use within the SDK.
 *
 * This will look at the users preferred shell and try to locate that within
 * the container environment.
 *
 * Returns: (transfer full): a [class@Dex.Future]
 */
DexFuture *
foundry_sdk_discover_shell (FoundrySdk *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_SDK (self));
  dex_return_error_if_fail (foundry_sdk_get_installed (self));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_sdk_discover_shell_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

/**
 * foundry_sdk_install:
 * @self: a [class@Foundry.Sdk]
 * @operation: a [class@Foundry.Operation]
 * @cancellable: (nullable): a [class@Dex.Cancellable]
 *
 * Installs an SDK.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a boolean.
 */
DexFuture *
foundry_sdk_install (FoundrySdk       *self,
                     FoundryOperation *operation,
                     DexCancellable   *cancellable)
{
  dex_return_error_if_fail (FOUNDRY_IS_SDK (self));
  dex_return_error_if_fail (FOUNDRY_IS_OPERATION (operation));
  dex_return_error_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (FOUNDRY_SDK_GET_CLASS (self)->install)
    return FOUNDRY_SDK_GET_CLASS (self)->install (self, operation, cancellable);

  return dex_future_new_true ();
}

/**
 * foundry_sdk_dup_config_option:
 * @self: a [class@Foundry.Sdk]
 *
 * Gets a config option that should be used as the default to work with
 * this particular SDK.
 *
 * Returns: (transfer full) (nullable): a string containing the config
 *    option or %NULL if unset.
 */
char *
foundry_sdk_dup_config_option (FoundrySdk             *self,
                               FoundrySdkConfigOption  option)
{
  g_return_val_if_fail (FOUNDRY_IS_SDK (self), NULL);

  if (FOUNDRY_SDK_GET_CLASS (self)->dup_config_option)
    return FOUNDRY_SDK_GET_CLASS (self)->dup_config_option (self, option);

  return NULL;
}

static DexFuture *
foundry_sdk_build_simple_fiber (FoundrySdk           *self,
                                FoundryBuildPipeline *pipeline,
                                const char * const   *argv)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  g_auto(GStrv) environ_ = NULL;

  g_assert (FOUNDRY_IS_SDK (self));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (argv && argv[0]);

  launcher = foundry_process_launcher_new ();

  if (!dex_await (foundry_sdk_prepare_to_build (self, pipeline, launcher, 0), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  /* Ensure PATH is applied if necessary */
  if (pipeline != NULL &&
      (config = foundry_build_pipeline_dup_config (pipeline)) &&
      (environ_ = foundry_config_dup_environ (config, FOUNDRY_LOCALITY_BUILD)))
    foundry_process_launcher_add_environ (launcher, (const char * const *)environ_);

  foundry_process_launcher_append_args (launcher, argv);

  if (!(subprocess = foundry_process_launcher_spawn_with_flags (launcher, G_SUBPROCESS_FLAGS_STDOUT_PIPE, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return foundry_subprocess_communicate_utf8 (subprocess, NULL);
}

/**
 * foundry_sdk_build_simple:
 * @self: a [class@Foundry.Sdk]
 * @pipeline: (nullable): a [class@Foundry.BuildPipeline]
 * @argv: the arguments to run
 *
 * This is a much simplified interface for [method@Foundry.Sdk.prepare_to_build]
 * for consumers that just want to run a simple command and get the stdout
 * output of the command.
 *
 * Use this when you want to quickly run something like `program --version`
 * when setting up the pipeline.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a UTF-8
 *   encoded string or rejects with error.
 *
 * Since: 1.1
 */
DexFuture *
foundry_sdk_build_simple (FoundrySdk           *self,
                          FoundryBuildPipeline *pipeline,
                          const char * const   *argv)
{
  dex_return_error_if_fail (FOUNDRY_IS_SDK (self));
  dex_return_error_if_fail (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  dex_return_error_if_fail (argv != NULL && argv[0] != NULL);

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_sdk_build_simple_fiber),
                                  3,
                                  FOUNDRY_TYPE_SDK, self,
                                  FOUNDRY_TYPE_BUILD_PIPELINE, pipeline,
                                  G_TYPE_STRV, argv);
}
