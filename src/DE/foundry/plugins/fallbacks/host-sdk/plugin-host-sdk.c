/* plugin-host-sdk.c
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

#include "plugin-host-sdk.h"

struct _PluginHostSdk
{
  FoundrySdk parent_instance;
  guint      in_flatpak : 1;
};

G_DEFINE_FINAL_TYPE (PluginHostSdk, plugin_host_sdk, FOUNDRY_TYPE_SDK)

static DexFuture *
plugin_host_sdk_prepare_to_build (FoundrySdk                *sdk,
                                  FoundryBuildPipeline      *pipeline,
                                  FoundryProcessLauncher    *launcher,
                                  FoundryBuildPipelinePhase  phase)
{
  g_assert (PLUGIN_IS_HOST_SDK (sdk));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  foundry_process_launcher_push_host (launcher);
  foundry_process_launcher_add_minimal_environment (launcher);

  return dex_future_new_true ();
}

static DexFuture *
plugin_host_sdk_prepare_to_run (FoundrySdk             *sdk,
                                FoundryBuildPipeline   *pipeline,
                                FoundryProcessLauncher *launcher)
{
  g_assert (PLUGIN_IS_HOST_SDK (sdk));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  foundry_process_launcher_push_host (launcher);
  foundry_process_launcher_add_minimal_environment (launcher);

  return dex_future_new_true ();
}

static void
plugin_host_sdk_class_init (PluginHostSdkClass *klass)
{
  FoundrySdkClass *sdk_class = FOUNDRY_SDK_CLASS (klass);

  sdk_class->prepare_to_build = plugin_host_sdk_prepare_to_build;
  sdk_class->prepare_to_run = plugin_host_sdk_prepare_to_run;
}

static void
plugin_host_sdk_init (PluginHostSdk *self)
{
  self->in_flatpak = g_file_test ("/.flatpak-info", G_FILE_TEST_EXISTS);
}

FoundrySdk *
plugin_host_sdk_new (FoundryContext *context)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);

  return g_object_new (PLUGIN_TYPE_HOST_SDK,
                       "context", context,
                       "id", "host",
                       "arch", foundry_get_default_arch (),
                       "name", _("My Computer"),
                       "kind", "host",
                       "installed", TRUE,
                       NULL);
}

char *
plugin_host_sdk_build_filename (PluginHostSdk  *self,
                                const char     *first_element,
                                ...)
{
  g_autofree char *joined = NULL;
  va_list args;

  va_start (args, first_element);
  joined = g_build_filename_valist (first_element, &args);
  va_end (args);

  if (self->in_flatpak)
    return g_build_filename ("/var/run/host", joined, NULL);

  if (g_path_is_absolute (joined))
    return g_steal_pointer (&joined);

  return g_build_filename ("/", joined, NULL);
}
