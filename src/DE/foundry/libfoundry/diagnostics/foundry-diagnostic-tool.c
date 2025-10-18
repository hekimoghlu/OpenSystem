/* foundry-diagnostic-tool.c
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

#include "foundry-build-manager.h"
#include "foundry-build-pipeline.h"
#include "foundry-command.h"
#include "foundry-diagnostic.h"
#include "foundry-diagnostic-tool.h"
#include "foundry-process-launcher.h"
#include "foundry-sdk-manager.h"
#include "foundry-sdk.h"
#include "foundry-subprocess.h"
#include "foundry-util.h"

typedef struct
{
  char **argv;
  char **environ;
} FoundryDiagnosticToolPrivate;

enum {
  PROP_0,
  PROP_ARGV,
  PROP_ENVIRON,
  N_PROPS
};

G_DEFINE_QUARK (foundry-diagnostic-tool-error, foundry_diagnostic_tool_error)
G_DEFINE_TYPE_WITH_PRIVATE (FoundryDiagnosticTool, foundry_diagnostic_tool, FOUNDRY_TYPE_DIAGNOSTIC_PROVIDER)

static GParamSpec *properties[N_PROPS];

static DexFuture *
foundry_diagnostic_tool_real_prepare (FoundryDiagnosticTool  *self,
                                      FoundryProcessLauncher *launcher,
                                      const char * const     *argv,
                                      const char * const     *environ,
                                      const char             *language)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundrySdkManager) sdk_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundrySdk) host = NULL;
  g_autoptr(FoundrySdk) no = NULL;
  g_autoptr(GError) error = NULL;
  gboolean prepared = FALSE;

  g_assert (FOUNDRY_IS_DIAGNOSTIC_TOOL (self));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  sdk_manager = foundry_context_dup_sdk_manager (context);
  host = dex_await_object (foundry_sdk_manager_find_by_id (sdk_manager, "host"), NULL);
  no = dex_await_object (foundry_sdk_manager_find_by_id (sdk_manager, "no"), NULL);

  build_manager = foundry_context_dup_build_manager (context);
  pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), NULL);

  /* If the command cannot be found in the pipeline then we will try to
   * fallback to locating the command on the host.
   */
  if (pipeline == NULL ||
      !dex_await (foundry_build_pipeline_contains_program (pipeline, argv[0]), NULL))
    {
      if (host != NULL)
        {
          if (dex_await (foundry_sdk_contains_program (host, argv[0]), NULL))
            {
              /* If we cannot find the program on the host, we must reject */
              if (!dex_await (foundry_sdk_prepare_to_build (host, NULL, launcher, FOUNDRY_BUILD_PIPELINE_PHASE_BUILD), &error))
                return dex_future_new_for_error (g_steal_pointer (&error));

              prepared = TRUE;
            }
        }
    }

  /* Now try the build pipeline if the tool is available in the build
   * environment.
   */
  if (!prepared && pipeline != NULL)
    {
      if (dex_await (foundry_build_pipeline_prepare (pipeline, launcher, FOUNDRY_BUILD_PIPELINE_PHASE_BUILD), NULL))
        prepared = TRUE;
    }

  /* As a last resort try things as a subprocess */
  if (!prepared && no != NULL &&
      dex_await (foundry_sdk_contains_program (no, argv[0]), NULL))
    prepared = TRUE;

  if (!prepared)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "Failed to locate command \"%s\"",
                                  argv[0]);

  foundry_process_launcher_append_args (launcher, argv);

  if (environ != NULL)
    foundry_process_launcher_add_environ (launcher, environ);

  return dex_future_new_true ();
}

static DexFuture *
foundry_diagnostic_tool_prepare (FoundryDiagnosticTool  *self,
                                 FoundryProcessLauncher *launcher,
                                 const char * const     *argv,
                                 const char * const     *environ,
                                 const char             *language)
{
  return FOUNDRY_DIAGNOSTIC_TOOL_GET_CLASS (self)->prepare (self, launcher, argv, environ, language);
}

static DexFuture *
foundry_diagnostic_tool_dup_bytes_for_stdin (FoundryDiagnosticTool *self,
                                             GFile                 *file,
                                             GBytes                *contents,
                                             const char            *language)
{
  if (FOUNDRY_DIAGNOSTIC_TOOL_GET_CLASS (self)->dup_bytes_for_stdin)
    return FOUNDRY_DIAGNOSTIC_TOOL_GET_CLASS (self)->dup_bytes_for_stdin (self, file, contents, language);

  return dex_future_new_take_boxed (G_TYPE_BYTES, g_bytes_new_static ("", 0));
}

static DexFuture *
foundry_diagnostic_tool_extract_from_stdout (FoundryDiagnosticTool *self,
                                             GFile                 *file,
                                             GBytes                *contents,
                                             const char            *language,
                                             GBytes                *stdout_bytes)
{
  if (FOUNDRY_DIAGNOSTIC_TOOL_GET_CLASS (self)->extract_from_stdout)
    return FOUNDRY_DIAGNOSTIC_TOOL_GET_CLASS (self)->extract_from_stdout (self, file, contents, language, stdout_bytes);

  return dex_future_new_take_object (g_list_store_new (FOUNDRY_TYPE_DIAGNOSTIC));
}

static DexFuture *
foundry_diagnostic_tool_diagnose_fiber (FoundryDiagnosticTool *self,
                                        FoundryContext        *context,
                                        const char * const    *argv,
                                        const char * const    *environ,
                                        GFile                 *file,
                                        GBytes                *contents,
                                        const char            *language)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GListModel) diagnostics = NULL;
  g_autoptr(GBytes) stdin_bytes = NULL;
  g_autoptr(GBytes) stdout_bytes = NULL;
  GSubprocessFlags flags = 0;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_DIAGNOSTIC_TOOL (self));
  g_assert (!file || G_IS_FILE (file));
  g_assert (file || contents);

  launcher = foundry_process_launcher_new ();

  if (!dex_await (foundry_diagnostic_tool_prepare (self, launcher, argv, environ, language), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  flags = G_SUBPROCESS_FLAGS_STDIN_PIPE | G_SUBPROCESS_FLAGS_STDOUT_PIPE | G_SUBPROCESS_FLAGS_STDERR_SILENCE;

  if (!(stdin_bytes = dex_await_boxed (foundry_diagnostic_tool_dup_bytes_for_stdin (self, file, contents, language), &error)))
    {
      flags &= ~G_SUBPROCESS_FLAGS_STDIN_PIPE;

      if (error != NULL)
        return dex_future_new_for_error (g_steal_pointer (&error));
    }

  if (!(subprocess = foundry_process_launcher_spawn_with_flags (launcher, flags, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(stdout_bytes = dex_await_boxed (foundry_subprocess_communicate (subprocess, stdin_bytes), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return foundry_diagnostic_tool_extract_from_stdout (self, file, contents, language, stdout_bytes);
}

static DexFuture *
foundry_diagnostic_tool_diagnose (FoundryDiagnosticProvider *provider,
                                  GFile                     *file,
                                  GBytes                    *contents,
                                  const char                *language)
{
  FoundryDiagnosticTool *self = (FoundryDiagnosticTool *)provider;
  FoundryDiagnosticToolPrivate *priv = foundry_diagnostic_tool_get_instance_private (self);
  g_autoptr(FoundryContext) context = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_DIAGNOSTIC_TOOL (self));
  dex_return_error_if_fail (!file || G_IS_FILE (file));
  dex_return_error_if_fail (file || contents);

  if (priv->argv == NULL || priv->argv[0] == NULL)
    return dex_future_new_reject (FOUNDRY_DIAGNOSTIC_TOOL_ERROR,
                                  FOUNDRY_DIAGNOSTIC_TOOL_ERROR_NO_COMMAND,
                                  "No command was provided");

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (provider));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_diagnostic_tool_diagnose_fiber),
                                  7,
                                  FOUNDRY_TYPE_DIAGNOSTIC_TOOL, self,
                                  FOUNDRY_TYPE_CONTEXT, context,
                                  G_TYPE_STRV, priv->argv,
                                  G_TYPE_STRV, priv->environ,
                                  G_TYPE_FILE, file,
                                  G_TYPE_BYTES, contents,
                                  G_TYPE_STRING, language);
}

static void
foundry_diagnostic_tool_finalize (GObject *object)
{
  FoundryDiagnosticTool *self = (FoundryDiagnosticTool *)object;
  FoundryDiagnosticToolPrivate *priv = foundry_diagnostic_tool_get_instance_private (self);

  g_clear_pointer (&priv->argv, g_strfreev);
  g_clear_pointer (&priv->environ, g_strfreev);

  G_OBJECT_CLASS (foundry_diagnostic_tool_parent_class)->finalize (object);
}

static void
foundry_diagnostic_tool_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  FoundryDiagnosticTool *self = FOUNDRY_DIAGNOSTIC_TOOL (object);

  switch (prop_id)
    {
    case PROP_ARGV:
      g_value_take_boxed (value, foundry_diagnostic_tool_dup_argv (self));
      break;

    case PROP_ENVIRON:
      g_value_take_boxed (value, foundry_diagnostic_tool_dup_environ (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_diagnostic_tool_set_property (GObject      *object,
                                      guint         prop_id,
                                      const GValue *value,
                                      GParamSpec   *pspec)
{
  FoundryDiagnosticTool *self = FOUNDRY_DIAGNOSTIC_TOOL (object);

  switch (prop_id)
    {
    case PROP_ARGV:
      foundry_diagnostic_tool_set_argv (self, g_value_get_boxed (value));
      break;

    case PROP_ENVIRON:
      foundry_diagnostic_tool_set_environ (self, g_value_get_boxed (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_diagnostic_tool_class_init (FoundryDiagnosticToolClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDiagnosticProviderClass *provider_class = FOUNDRY_DIAGNOSTIC_PROVIDER_CLASS (klass);

  object_class->finalize = foundry_diagnostic_tool_finalize;
  object_class->get_property = foundry_diagnostic_tool_get_property;
  object_class->set_property = foundry_diagnostic_tool_set_property;

  provider_class->diagnose = foundry_diagnostic_tool_diagnose;

  klass->prepare = foundry_diagnostic_tool_real_prepare;

  properties[PROP_ARGV] =
    g_param_spec_boxed ("argv", NULL, NULL,
                        G_TYPE_STRV,
                        (G_PARAM_READWRITE |
                         G_PARAM_EXPLICIT_NOTIFY |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_ENVIRON] =
    g_param_spec_boxed ("environ", NULL, NULL,
                        G_TYPE_STRV,
                        (G_PARAM_READWRITE |
                         G_PARAM_EXPLICIT_NOTIFY |
                         G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_diagnostic_tool_init (FoundryDiagnosticTool *self)
{
}

/**
 * foundry_diagnostic_tool_dup_argv:
 * @self: a [class@Foundry.DiagnosticTool]
 *
 * Returns: (transfer full) (nullable):
 */
char **
foundry_diagnostic_tool_dup_argv (FoundryDiagnosticTool *self)
{
  FoundryDiagnosticToolPrivate *priv = foundry_diagnostic_tool_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC_TOOL (self), NULL);

  return g_strdupv (priv->argv);
}

void
foundry_diagnostic_tool_set_argv (FoundryDiagnosticTool *self,
                                  const char * const    *argv)
{
  FoundryDiagnosticToolPrivate *priv = foundry_diagnostic_tool_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_DIAGNOSTIC_TOOL (self));

  if (foundry_set_strv (&priv->argv, argv))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ARGV]);
}

/**
 * foundry_diagnostic_tool_dup_environ:
 * @self: a [class@Foundry.DiagnosticTool]
 *
 * Returns: (transfer full) (nullable):
 */
char **
foundry_diagnostic_tool_dup_environ (FoundryDiagnosticTool *self)
{
  FoundryDiagnosticToolPrivate *priv = foundry_diagnostic_tool_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTIC_TOOL (self), NULL);

  return g_strdupv (priv->environ);
}

void
foundry_diagnostic_tool_set_environ (FoundryDiagnosticTool *self,
                                     const char * const    *environ)
{
  FoundryDiagnosticToolPrivate *priv = foundry_diagnostic_tool_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_DIAGNOSTIC_TOOL (self));

  if (foundry_set_strv (&priv->environ, environ))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ENVIRON]);
}
