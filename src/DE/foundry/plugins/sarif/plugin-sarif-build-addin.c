/* plugin-sarif-build-addin.c
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

#include <glib/gi18n-lib.h>

#include "plugin-sarif-build-addin.h"
#include "plugin-sarif-build-stage.h"
#include "plugin-sarif-service.h"

struct _PluginSarifBuildAddin
{
  FoundryBuildAddin      parent_instance;
  PluginSarifBuildStage *stage;
};

G_DEFINE_FINAL_TYPE (PluginSarifBuildAddin, plugin_sarif_build_addin, FOUNDRY_TYPE_BUILD_ADDIN)

static GRegex *version_regex;

static void
plugin_sarif_build_addin_setup (PluginSarifBuildAddin *self,
                                FoundryBuildPipeline  *pipeline)
{
  g_autoptr(PluginSarifService) service = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autofree char *address = NULL;

  g_assert (PLUGIN_IS_SARIF_BUILD_ADDIN (self));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  service = foundry_context_dup_service_typed (context, PLUGIN_TYPE_SARIF_SERVICE);
  address = dex_await_string (plugin_sarif_service_socket_path (service), NULL);

  if (address != NULL)
    foundry_build_pipeline_setenv (pipeline, "SARIF_SOCKET", address);
}

static DexFuture *
plugin_sarif_build_addin_load_fiber (gpointer data)
{
  PluginSarifBuildAddin *self = data;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundrySdk) sdk = NULL;
  g_autofree char *stdout_buf = NULL;

  g_assert (PLUGIN_IS_SARIF_BUILD_ADDIN (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  if (!(pipeline = foundry_build_addin_dup_pipeline (FOUNDRY_BUILD_ADDIN (self))) ||
      !(sdk = foundry_build_pipeline_dup_sdk (pipeline)))
    return dex_future_new_true ();

  /* Sniff the GCC version available in the SDK and if it is new enough
   * then we will set an environment variable to redirect SARIF output
   * to the appropriate socket.
   */

  if (dex_await (foundry_sdk_contains_program (sdk, "gcc"), NULL) &&
      (stdout_buf = dex_await_string (foundry_sdk_build_simple (sdk,
                                                                pipeline,
                                                                FOUNDRY_STRV_INIT ("gcc", "--version")),
                                      NULL)))
    {
      g_autoptr(GMatchInfo) match_info = NULL;

      if (g_regex_match_full (version_regex, stdout_buf, -1, 0, 0, &match_info, NULL))
        {
          g_autofree char *version = g_match_info_fetch (match_info, 1);
          int major;

          g_debug ("GCC version %s detected", version);

          *strchr (version, '.') = 0;
          major = atoi (version);

          if (major >= 16)
            plugin_sarif_build_addin_setup (self, pipeline);
        }
    }

  self->stage = g_object_new (PLUGIN_TYPE_SARIF_BUILD_STAGE,
                              "context", context,
                              "kind", "sarif",
                              "title", _("Clear existing diagnostics"),
                              NULL);

  foundry_build_pipeline_add_stage (pipeline, FOUNDRY_BUILD_STAGE (self->stage));

  return dex_future_new_true ();
}

static DexFuture *
plugin_sarif_build_addin_load (FoundryBuildAddin *addin)
{
  g_assert (PLUGIN_IS_SARIF_BUILD_ADDIN (addin));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_sarif_build_addin_load_fiber,
                              g_object_ref (addin),
                              g_object_unref);
}

static DexFuture *
plugin_sarif_build_addin_unload (FoundryBuildAddin *addin)
{
  PluginSarifBuildAddin *self = (PluginSarifBuildAddin *)addin;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;

  g_assert (PLUGIN_IS_SARIF_BUILD_ADDIN (addin));

  pipeline = foundry_build_addin_dup_pipeline (FOUNDRY_BUILD_ADDIN (self));

  if (self->stage != NULL)
    {
      foundry_build_pipeline_remove_stage (pipeline, FOUNDRY_BUILD_STAGE (self->stage));
      g_clear_object (&self->stage);
    }

  foundry_build_pipeline_setenv (pipeline, "SARIF_SOCKET", NULL);

  return dex_future_new_true ();
}

static void
plugin_sarif_build_addin_class_init (PluginSarifBuildAddinClass *klass)
{
  FoundryBuildAddinClass *build_addin_class = FOUNDRY_BUILD_ADDIN_CLASS (klass);

  build_addin_class->load = plugin_sarif_build_addin_load;
  build_addin_class->unload = plugin_sarif_build_addin_unload;
}

static void
plugin_sarif_build_addin_init (PluginSarifBuildAddin *self)
{
  if (version_regex == NULL)
    version_regex = g_regex_new ("(\\d+\\.\\d+\\.\\d+)", G_REGEX_OPTIMIZE, 0, NULL);
}
