/* plugin-flatpak-dependency-provider.c
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

#include "plugin-flatpak-config.h"
#include "plugin-flatpak-dependency.h"
#include "plugin-flatpak-dependency-provider.h"
#include "plugin-flatpak-util.h"

struct _PluginFlatpakDependencyProvider
{
  FoundryDependencyProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginFlatpakDependencyProvider, plugin_flatpak_dependency_provider, FOUNDRY_TYPE_DEPENDENCY_PROVIDER)

static DexFuture *
plugin_flatpak_dependency_provider_list_dependencies (FoundryDependencyProvider *dependency_provider,
                                                      FoundryConfig             *config,
                                                      FoundryDependency         *parent)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListStore) store = NULL;

  g_assert (PLUGIN_IS_FLATPAK_DEPENDENCY_PROVIDER (dependency_provider));
  g_assert (FOUNDRY_IS_CONFIG (config));
  g_assert (!parent || FOUNDRY_IS_DEPENDENCY (parent));

  dex_return_error_if_fail (!parent || PLUGIN_IS_FLATPAK_DEPENDENCY (parent));

  store = g_list_store_new (FOUNDRY_TYPE_DEPENDENCY);
  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (config));

  /* TODO: Probably want to check the SDK and include that as a dependency
   * of the config (so that we can provide update API for that later on).
   */

  if (PLUGIN_IS_FLATPAK_CONFIG (config))
    {
      g_autoptr(FoundryFlatpakManifest) manifest = plugin_flatpak_config_dup_manifest (PLUGIN_FLATPAK_CONFIG (config));
      g_autoptr(FoundryFlatpakModules) modules = foundry_flatpak_manifest_dup_modules (manifest);
      g_autoptr(FoundryFlatpakModule) primary_module = plugin_flatpak_config_dup_primary_module (PLUGIN_FLATPAK_CONFIG (config));
      guint n_items = 0;

      if (modules != NULL)
        n_items = g_list_model_get_n_items (G_LIST_MODEL (modules));

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryFlatpakModule) module = g_list_model_get_item (G_LIST_MODEL (modules), i);
          g_autoptr(PluginFlatpakDependency) dependency = plugin_flatpak_dependency_new (context, dependency_provider, module);

          if (primary_module != module)
            g_list_store_append (store, dependency);
        }
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static DexFuture *
plugin_flatpak_dependency_provider_update_dependencies (FoundryDependencyProvider *provider,
                                                        FoundryConfig             *config,
                                                        GListModel                *dependencies,
                                                        int                        pty_fd,
                                                        DexCancellable            *cancellable)
{
  PluginFlatpakDependencyProvider *self = (PluginFlatpakDependencyProvider *)provider;
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) file = NULL;
  g_autofree char *state_dir = NULL;

  g_assert (PLUGIN_IS_FLATPAK_DEPENDENCY_PROVIDER (self));
  g_assert (FOUNDRY_IS_CONFIG (config));
  g_assert (G_IS_LIST_MODEL (dependencies));
  g_assert (pty_fd >= -1);
  g_assert (DEX_IS_CANCELLABLE (cancellable));

  if (!PLUGIN_IS_FLATPAK_CONFIG (config))
    return dex_future_new_true ();

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (provider));
  state_dir = plugin_flatpak_dup_state_dir (context);
  file = plugin_flatpak_config_dup_file (PLUGIN_FLATPAK_CONFIG (config));

  /* flatpak-builder should be bundled when using Foundry, so we can execute
   * it as a subprocess even if it then jumps out to the host for
   * "flatpak build" command.
   */
  launcher = foundry_process_launcher_new ();
  foundry_process_launcher_append_argv (launcher, "flatpak-builder");
  foundry_process_launcher_append_argv (launcher, "--state-dir");
  foundry_process_launcher_append_argv (launcher, state_dir);
  foundry_process_launcher_append_argv (launcher, "--download-only");
  /* The staging-directory isn't needed (and would require a pipeline to
   * be setup) which we don't want to have to require. So just use something
   * fake which wont get used anyway.
   */
  foundry_process_launcher_append_argv (launcher, "__tmp");
  foundry_process_launcher_append_argv (launcher, g_file_peek_path (file));

  foundry_process_launcher_take_fd (launcher, dup (pty_fd), STDIN_FILENO);
  foundry_process_launcher_take_fd (launcher, dup (pty_fd), STDOUT_FILENO);
  foundry_process_launcher_take_fd (launcher, dup (pty_fd), STDERR_FILENO);

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_subprocess_wait_check (subprocess);
}

static void
plugin_flatpak_dependency_provider_class_init (PluginFlatpakDependencyProviderClass *klass)
{
  FoundryDependencyProviderClass *dependency_provider_class = FOUNDRY_DEPENDENCY_PROVIDER_CLASS (klass);

  dependency_provider_class->list_dependencies = plugin_flatpak_dependency_provider_list_dependencies;
  dependency_provider_class->update_dependencies = plugin_flatpak_dependency_provider_update_dependencies;
}

static void
plugin_flatpak_dependency_provider_init (PluginFlatpakDependencyProvider *self)
{
}
