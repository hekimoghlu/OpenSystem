/* plugin-jhbuild-sdk-provider.c
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

#include <libdex.h>

#include "plugin-jhbuild-sdk-provider.h"
#include "plugin-jhbuild-sdk.h"

struct _PluginJhbuildSdkProvider
{
  FoundrySdkProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginJhbuildSdkProvider, plugin_jhbuild_sdk_provider, FOUNDRY_TYPE_SDK_PROVIDER)

static DexFuture *
query_envvar (const char *envvar)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (envvar != NULL);

  launcher = foundry_process_launcher_new ();

  foundry_process_launcher_push_host (launcher);
  foundry_process_launcher_append_argv (launcher, "jhbuild");
  foundry_process_launcher_append_argv (launcher, "run");
  foundry_process_launcher_append_argv (launcher, "sh");
  foundry_process_launcher_append_argv (launcher, "-c");
  foundry_process_launcher_append_formatted (launcher, "echo -n $%s", envvar);

  if (!(subprocess = foundry_process_launcher_spawn_with_flags (launcher, G_SUBPROCESS_FLAGS_STDOUT_PIPE, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return foundry_subprocess_communicate_utf8 (subprocess, NULL);
}

static DexFuture *
plugin_jhbuild_sdk_provider_load_fiber (gpointer data)
{
  PluginJhbuildSdkProvider *self = data;
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundrySdk) sdk = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *prefix = NULL;
  g_autofree char *libdir = NULL;

  g_assert (PLUGIN_IS_JHBUILD_SDK_PROVIDER (self));

  if (!(context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    return dex_future_new_true ();

  launcher = foundry_process_launcher_new ();
  foundry_process_launcher_push_host (launcher);
  foundry_process_launcher_append_argv (launcher, "which");
  foundry_process_launcher_append_argv (launcher, "jhbuild");
  foundry_process_launcher_take_fd (launcher, -1, STDIN_FILENO);
  foundry_process_launcher_take_fd (launcher, -1, STDOUT_FILENO);
  foundry_process_launcher_take_fd (launcher, -1, STDERR_FILENO);

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!dex_await (dex_subprocess_wait_check (subprocess), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  prefix = dex_await_string (query_envvar ("JHBUILD_PREFIX"), NULL);
  libdir = dex_await_string (query_envvar ("JHBUILD_LIBDIR"), NULL);
  sdk = plugin_jhbuild_sdk_new (context, prefix, libdir);

  foundry_sdk_provider_sdk_added (FOUNDRY_SDK_PROVIDER (self), sdk);

  return dex_future_new_true ();
}

static DexFuture *
plugin_jhbuild_sdk_provider_load (FoundrySdkProvider *sdk_provider)
{
  return dex_scheduler_spawn (NULL, 0,
                              plugin_jhbuild_sdk_provider_load_fiber,
                              g_object_ref (sdk_provider),
                              g_object_unref);
}

static void
plugin_jhbuild_sdk_provider_class_init (PluginJhbuildSdkProviderClass *klass)
{
  FoundrySdkProviderClass *sdk_provider_class = FOUNDRY_SDK_PROVIDER_CLASS (klass);

  sdk_provider_class->load = plugin_jhbuild_sdk_provider_load;
}

static void
plugin_jhbuild_sdk_provider_init (PluginJhbuildSdkProvider *self)
{
}
