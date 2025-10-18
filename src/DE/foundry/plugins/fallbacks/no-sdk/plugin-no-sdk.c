/* plugin-no-sdk.c
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

#include "plugin-no-sdk.h"

struct _PluginNoSdk
{
  FoundrySdk parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginNoSdk, plugin_no_sdk, FOUNDRY_TYPE_SDK)

static DexFuture *
plugin_no_sdk_prepare_to_build (FoundrySdk                *sdk,
                                FoundryBuildPipeline      *pipeline,
                                FoundryProcessLauncher    *launcher,
                                FoundryBuildPipelinePhase  phase)
{
  g_assert (PLUGIN_IS_NO_SDK (sdk));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  return dex_future_new_true ();
}

static DexFuture *
plugin_no_sdk_prepare_to_run (FoundrySdk             *sdk,
                              FoundryBuildPipeline   *pipeline,
                              FoundryProcessLauncher *launcher)
{
  g_assert (PLUGIN_IS_NO_SDK (sdk));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  return dex_future_new_true ();
}

static void
plugin_no_sdk_class_init (PluginNoSdkClass *klass)
{
  FoundrySdkClass *sdk_class = FOUNDRY_SDK_CLASS (klass);

  sdk_class->prepare_to_build = plugin_no_sdk_prepare_to_build;
  sdk_class->prepare_to_run = plugin_no_sdk_prepare_to_run;
}

static void
plugin_no_sdk_init (PluginNoSdk *self)
{
}

FoundrySdk *
plugin_no_sdk_new (FoundryContext *context)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);

  return g_object_new (PLUGIN_TYPE_NO_SDK,
                       "context", context,
                       "id", "no",
                       "arch", foundry_get_default_arch (),
                       "name", _("No SDK"),
                       "installed", TRUE,
                       "kind", "host",
                       NULL);
}
