/* plugin-flatpak-config-provider.c
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

#include "plugin-flatpak-config-provider.h"
#include "plugin-flatpak-config.h"

#define DISCOVERY_MAX_DEPTH 3

struct _PluginFlatpakConfigProvider
{
  FoundryConfigProvider parent_instnace;
};

G_DEFINE_FINAL_TYPE (PluginFlatpakConfigProvider, plugin_flatpak_config_provider, FOUNDRY_TYPE_CONFIG_PROVIDER)

static GRegex *filename_regex;

static DexFuture *
plugin_flatpak_config_provider_load_fiber (gpointer user_data)
{
  PluginFlatpakConfigProvider *self = user_data;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) matching = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) project_dir = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_FLATPAK_CONFIG_PROVIDER (self));

  if (!(context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))) ||
      !(project_dir = foundry_context_dup_project_directory (context)))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_CANCELLED,
                                  "Operation cancelled");

  /* First find all the files that match potential flatpak manifests */
  if (!(matching = dex_await_boxed (foundry_file_find_regex_with_depth (project_dir,
                                                                        filename_regex,
                                                                        DISCOVERY_MAX_DEPTH),
                                    &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  for (guint i = 0; i < matching->len; i++)
    {
      g_autoptr(FoundryFlatpakManifestLoader) loader = NULL;
      g_autoptr(FoundryFlatpakManifest) manifest = NULL;
      g_autoptr(PluginFlatpakConfig) config = NULL;
      g_autoptr(GError) manifest_error = NULL;
      GFile *match = g_ptr_array_index (matching, i);

      loader = foundry_flatpak_manifest_loader_new (match);

      if (!(manifest = dex_await_object (foundry_flatpak_manifest_loader_load (loader), &manifest_error)))
        {
          FOUNDRY_CONTEXTUAL_DEBUG (self,
                                    "Ignoring file \"%s\" because error: %s",
                                    g_file_peek_path (match),
                                    manifest_error->message);
          continue;
        }

      config = plugin_flatpak_config_new (context, manifest, match);
      foundry_config_provider_config_added (FOUNDRY_CONFIG_PROVIDER (self),
                                            FOUNDRY_CONFIG (config));
    }

  return dex_future_new_true ();
}


static DexFuture *
plugin_flatpak_config_provider_load (FoundryConfigProvider *provider)
{
  PluginFlatpakConfigProvider *self = (PluginFlatpakConfigProvider *)provider;
  DexFuture *future;

  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_FLATPAK_CONFIG_PROVIDER (provider));

  future = dex_scheduler_spawn (NULL, 0,
                                plugin_flatpak_config_provider_load_fiber,
                                g_object_ref (self),
                                g_object_unref);

  FOUNDRY_RETURN (future);
}

static void
plugin_flatpak_config_provider_class_init (PluginFlatpakConfigProviderClass *klass)
{
  FoundryConfigProviderClass *config_provider_class = FOUNDRY_CONFIG_PROVIDER_CLASS (klass);
  g_autoptr(GError) error = NULL;

  config_provider_class->load = plugin_flatpak_config_provider_load;

  /* Something that looks like an application ID with a json, yml, or yaml
   * filename suffix. We try to encode some basic rules of the application
   * id to reduce the chances we get something that cannot match.
   */
  filename_regex = g_regex_new ("([A-Za-z][A-Za-z0-9\\-_]*)(\\.([A-Za-z][A-Za-z0-9\\-_]*))+\\.(json|yml|yaml)",
                                G_REGEX_OPTIMIZE,
                                G_REGEX_MATCH_DEFAULT,
                                &error);
  g_assert_no_error (error);
}

static void
plugin_flatpak_config_provider_init (PluginFlatpakConfigProvider *self)
{
}
