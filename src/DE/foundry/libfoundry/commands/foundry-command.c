/* foundry-command.c
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
#include "foundry-command-provider.h"
#include "foundry-command-private.h"
#include "foundry-debug.h"
#include "foundry-process-launcher.h"
#include "foundry-util.h"

typedef struct
{
  GWeakRef provider_wr;
  char **argv;
  char **environ;
  char *id;
  char *cwd;
  char *name;
  FoundryCommandLocality locality : 3;
} FoundryCommandPrivate;

enum {
  PROP_0,
  PROP_ARGV,
  PROP_CWD,
  PROP_ENVIRON,
  PROP_ID,
  PROP_LOCALITY,
  PROP_NAME,
  PROP_PROVIDER,
  N_PROPS
};

G_DEFINE_TYPE_WITH_PRIVATE (FoundryCommand, foundry_command, FOUNDRY_TYPE_CONTEXTUAL)

static GParamSpec *properties[N_PROPS];

static DexFuture *
foundry_command_prepare_fiber (FoundryBuildPipeline      *pipeline,
                               FoundryProcessLauncher    *launcher,
                               FoundryContext            *context,
                               const char                *cwd,
                               const char * const        *argv,
                               const char * const        *extra_environ,
                               FoundryBuildPipelinePhase  phase,
                               FoundryCommandLocality     locality)
{
  g_autoptr(GFile) srcdir = NULL;
  g_autoptr(GError) error = NULL;
  g_auto(GStrv) environ = NULL;
  g_autofree char *path = NULL;

  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  switch (locality)
    {
    case FOUNDRY_COMMAND_LOCALITY_SUBPROCESS:
      break;

    case FOUNDRY_COMMAND_LOCALITY_HOST:
      foundry_process_launcher_push_host (launcher);
      foundry_process_launcher_add_minimal_environment (launcher);
      break;

    case FOUNDRY_COMMAND_LOCALITY_BUILD_PIPELINE:
      if (pipeline == NULL)
        return dex_future_new_reject (G_IO_ERROR,
                                      G_IO_ERROR_FAILED,
                                      "Command requires a build pipeline but none was provided");

      if (!dex_await (foundry_build_pipeline_prepare (pipeline, launcher, phase), &error))
        return dex_future_new_for_error (g_steal_pointer (&error));

      break;

    case FOUNDRY_COMMAND_LOCALITY_APPLICATION:
      if (pipeline == NULL)
        return dex_future_new_reject (G_IO_ERROR,
                                      G_IO_ERROR_FAILED,
                                      "Command requires a build pipeline but none was provided");

      if (!dex_await (foundry_build_pipeline_prepare_for_run (pipeline, launcher), &error))
        return dex_future_new_for_error (g_steal_pointer (&error));

      break;

    case FOUNDRY_COMMAND_LOCALITY_LAST:
    default:
      g_assert_not_reached ();
    }

  if ((srcdir = foundry_context_dup_project_directory (context)) &&
      g_file_is_native (srcdir) &&
      (path = g_file_get_path (srcdir)))
    environ = g_environ_setenv (environ, "SRCDIR", path, TRUE);

  if (pipeline != NULL)
    {
      g_autofree char *builddir = foundry_build_pipeline_dup_builddir (pipeline);
      g_autofree char *arch = foundry_build_pipeline_dup_arch (pipeline);

      if (builddir != NULL)
        environ = g_environ_setenv (environ, "BUILDDIR", builddir, TRUE);

      if (arch != NULL)
        environ = g_environ_setenv (environ, "FOUNDRY_PIPELINE_ARCH", arch, TRUE);
    }

  if (environ != NULL)
    foundry_process_launcher_push_expansion (launcher, (const char * const *)environ);

  if (cwd != NULL)
    foundry_process_launcher_set_cwd (launcher, cwd);

  if (argv != NULL)
    foundry_process_launcher_set_argv (launcher, argv);

  if (extra_environ != NULL)
    foundry_process_launcher_add_environ (launcher, extra_environ);

  return dex_future_new_true ();
}

static DexFuture *
foundry_command_real_prepare (FoundryCommand            *command,
                              FoundryBuildPipeline      *pipeline,
                              FoundryProcessLauncher    *launcher,
                              FoundryBuildPipelinePhase  phase)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (command);
  g_autoptr(FoundryContext) context = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_COMMAND (command));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (command));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_command_prepare_fiber),
                                  8,
                                  FOUNDRY_TYPE_BUILD_PIPELINE, pipeline,
                                  FOUNDRY_TYPE_PROCESS_LAUNCHER, launcher,
                                  FOUNDRY_TYPE_CONTEXT, context,
                                  G_TYPE_STRING, priv->cwd,
                                  G_TYPE_STRV, priv->argv,
                                  G_TYPE_STRV, priv->environ,
                                  FOUNDRY_TYPE_BUILD_PIPELINE_PHASE, phase,
                                  FOUNDRY_TYPE_COMMAND_LOCALITY, priv->locality);

}

static void
foundry_command_finalize (GObject *object)
{
  FoundryCommand *self = (FoundryCommand *)object;
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  g_weak_ref_clear (&priv->provider_wr);

  g_clear_pointer (&priv->argv, g_strfreev);
  g_clear_pointer (&priv->environ, g_strfreev);
  g_clear_pointer (&priv->cwd, g_free);
  g_clear_pointer (&priv->id, g_free);
  g_clear_pointer (&priv->name, g_free);

  G_OBJECT_CLASS (foundry_command_parent_class)->finalize (object);
}

static void
foundry_command_get_property (GObject    *object,
                              guint       prop_id,
                              GValue     *value,
                              GParamSpec *pspec)
{
  FoundryCommand *self = FOUNDRY_COMMAND (object);

  switch (prop_id)
    {
    case PROP_ARGV:
      g_value_take_boxed (value, foundry_command_dup_argv (self));
      break;

    case PROP_CWD:
      g_value_take_string (value, foundry_command_dup_cwd (self));
      break;

    case PROP_ENVIRON:
      g_value_take_boxed (value, foundry_command_dup_environ (self));
      break;

    case PROP_ID:
      g_value_take_string (value, foundry_command_dup_id (self));
      break;

    case PROP_LOCALITY:
      g_value_set_enum (value, foundry_command_get_locality (self));
      break;

    case PROP_NAME:
      g_value_take_string (value, foundry_command_dup_name (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_command_set_property (GObject      *object,
                              guint         prop_id,
                              const GValue *value,
                              GParamSpec   *pspec)
{
  FoundryCommand *self = FOUNDRY_COMMAND (object);

  switch (prop_id)
    {
    case PROP_ARGV:
      foundry_command_set_argv (self, g_value_get_boxed (value));
      break;

    case PROP_CWD:
      foundry_command_set_cwd (self, g_value_get_string (value));
      break;

    case PROP_ENVIRON:
      foundry_command_set_environ (self, g_value_get_boxed (value));
      break;

    case PROP_ID:
      foundry_command_set_id (self, g_value_get_string (value));
      break;

    case PROP_LOCALITY:
      foundry_command_set_locality (self, g_value_get_enum (value));
      break;

    case PROP_NAME:
      foundry_command_set_name (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_command_class_init (FoundryCommandClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_command_finalize;
  object_class->get_property = foundry_command_get_property;
  object_class->set_property = foundry_command_set_property;

  klass->prepare = foundry_command_real_prepare;

  properties[PROP_ARGV] =
    g_param_spec_boxed ("argv", NULL, NULL,
                        G_TYPE_STRV,
                        (G_PARAM_READWRITE |
                         G_PARAM_EXPLICIT_NOTIFY |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_CWD] =
    g_param_spec_string ("cwd", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ENVIRON] =
    g_param_spec_boxed ("environ", NULL, NULL,
                        G_TYPE_STRV,
                        (G_PARAM_READWRITE |
                         G_PARAM_EXPLICIT_NOTIFY |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_LOCALITY] =
    g_param_spec_enum ("locality", NULL, NULL,
                       FOUNDRY_TYPE_COMMAND_LOCALITY,
                       FOUNDRY_COMMAND_LOCALITY_BUILD_PIPELINE,
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
                         FOUNDRY_TYPE_COMMAND_PROVIDER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_command_init (FoundryCommand *self)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  priv->locality = FOUNDRY_COMMAND_LOCALITY_BUILD_PIPELINE;
}

/**
 * foundry_command_dup_argv:
 * @self: a [class@Foundry.Command]
 *
 * Returns: (transfer full) (nullable): a string array of arguments for
 *   the command to run.
 */
char **
foundry_command_dup_argv (FoundryCommand *self)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_COMMAND (self), NULL);

  return g_strdupv (priv->argv);
}

void
foundry_command_set_argv (FoundryCommand     *self,
                          const char * const *argv)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);
  char **copy;

  g_return_if_fail (FOUNDRY_IS_COMMAND (self));

  if (argv == (const char * const *)priv->argv)
    return;

  copy = g_strdupv ((char **)argv);
  g_strfreev (priv->argv);
  priv->argv = copy;

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ARGV]);
}

/**
 * foundry_command_dup_environ:
 * @self: a [class@Foundry.Command]
 *
 * Returns: (transfer full) (nullable): a string array containing the
 *   environment of %NULL.
 */
char **
foundry_command_dup_environ (FoundryCommand *self)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_COMMAND (self), NULL);

  return g_strdupv (priv->environ);
}

void
foundry_command_set_environ (FoundryCommand     *self,
                             const char * const *environ)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);
  char **copy;

  g_return_if_fail (FOUNDRY_IS_COMMAND (self));

  if (environ == (const char * const *)priv->environ)
    return;

  copy = g_strdupv ((char **)environ);
  g_strfreev (priv->environ);
  priv->environ = copy;

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ENVIRON]);
}

char *
foundry_command_dup_cwd (FoundryCommand *self)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_COMMAND (self), NULL);

  return g_strdup (priv->cwd);
}

void
foundry_command_set_cwd (FoundryCommand *self,
                         const char     *cwd)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_COMMAND (self));

  if (g_set_str (&priv->cwd, cwd))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CWD]);
}

char *
foundry_command_dup_id (FoundryCommand *self)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_COMMAND (self), NULL);

  return g_strdup (priv->id);
}

void
foundry_command_set_id (FoundryCommand *self,
                        const char     *id)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_COMMAND (self));

  if (g_set_str (&priv->id, id))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ID]);
}

char *
foundry_command_dup_name (FoundryCommand *self)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_COMMAND (self), NULL);

  return g_strdup (priv->name);
}

void
foundry_command_set_name (FoundryCommand *self,
                          const char     *name)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_COMMAND (self));

  if (g_set_str (&priv->name, name))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_NAME]);
}

/**
 * foundry_command_can_default:
 * @self: a [class@Foundry.Command]
 * @priority: (out) (nullable): a location for the priority, or %NULL
 *
 * Checks to see if @self is suitable to be run as the default command when
 * running a project. The priority indicates if it should take priority over
 * other commands which can be default. The highest priority value wins.
 *
 * Returns: %TRUE if @self can be the default command
 */
gboolean
foundry_command_can_default (FoundryCommand *self,
                             guint          *priority)
{
  guint local;

  g_return_val_if_fail (FOUNDRY_IS_COMMAND (self), FALSE);

  if (priority == NULL)
    priority = &local;

  if (FOUNDRY_COMMAND_GET_CLASS (self)->can_default)
    return FOUNDRY_COMMAND_GET_CLASS (self)->can_default (self, priority);

  return FALSE;
}

/**
 * foundry_command_prepare:
 * @self: a [class@Foundry.Command]
 * @pipeline: (nullable): an optional [class@Foundry.BuildPipeline]
 * @launcher: a [class@Foundry.ProcessLauncher]
 * @phase: the phase of the pipeline.
 *
 * Prepares @launcher to run @self.
 *
 * If @pipeline is set, the command may use that to run the command within
 * a particular environment based on the locality settings.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any value
 *   when the preparation has completed. Otherwise rejects with error.
 */
DexFuture *
foundry_command_prepare (FoundryCommand            *self,
                         FoundryBuildPipeline      *pipeline,
                         FoundryProcessLauncher    *launcher,
                         FoundryBuildPipelinePhase  phase)
{
  dex_return_error_if_fail (FOUNDRY_IS_COMMAND (self));
  dex_return_error_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));
  dex_return_error_if_fail (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));

  return FOUNDRY_COMMAND_GET_CLASS (self)->prepare (self, pipeline, launcher, phase);
}

FoundryCommandProvider *
_foundry_command_dup_provider (FoundryCommand *self)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_COMMAND (self), NULL);

  return g_weak_ref_get (&priv->provider_wr);
}

void
_foundry_command_set_provider (FoundryCommand         *self,
                               FoundryCommandProvider *provider)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);
  g_autoptr(FoundryCommandProvider) previous = NULL;

  g_return_if_fail (FOUNDRY_IS_COMMAND (self));
  g_return_if_fail (!provider || FOUNDRY_IS_COMMAND_PROVIDER (provider));

  previous = g_weak_ref_get (&priv->provider_wr);

  if (previous == provider)
    return;

  g_weak_ref_set (&priv->provider_wr, provider);
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_PROVIDER]);
}

FoundryCommandLocality
foundry_command_get_locality (FoundryCommand *self)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_COMMAND (self), 0);

  return priv->locality;
}

void
foundry_command_set_locality (FoundryCommand         *self,
                              FoundryCommandLocality  locality)
{
  FoundryCommandPrivate *priv = foundry_command_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_COMMAND (self));
  g_return_if_fail (locality < FOUNDRY_COMMAND_LOCALITY_LAST);

  if (locality != priv->locality)
    {
      priv->locality = locality;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_LOCALITY]);
    }
}

G_DEFINE_ENUM_TYPE (FoundryCommandLocality, foundry_command_locality,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_COMMAND_LOCALITY_SUBPROCESS, "subprocess"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_COMMAND_LOCALITY_HOST, "host"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_COMMAND_LOCALITY_BUILD_PIPELINE, "pipeline"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_COMMAND_LOCALITY_APPLICATION, "application"))

FoundryCommand *
foundry_command_new (FoundryContext *context)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);

  return g_object_new (FOUNDRY_TYPE_COMMAND,
                       "context", context,
                       NULL);
}
