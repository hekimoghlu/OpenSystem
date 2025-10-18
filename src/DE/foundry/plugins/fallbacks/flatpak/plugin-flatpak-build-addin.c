/* plugin-flatpak-build-addin.c
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

#include "plugin-flatpak-autogen-stage.h"
#include "plugin-flatpak-build-addin.h"
#include "plugin-flatpak-bundle-stage.h"
#include "plugin-flatpak-commit-stage.h"
#include "plugin-flatpak-config.h"
#include "plugin-flatpak-dependencies-stage.h"
#include "plugin-flatpak-download-stage.h"
#include "plugin-flatpak-export-stage.h"
#include "plugin-flatpak-prepare-stage.h"
#include "plugin-flatpak-simple-stage.h"
#include "plugin-flatpak-util.h"

struct _PluginFlatpakBuildAddin
{
  FoundryBuildAddin  parent_instance;
  FoundryBuildStage *autogen;
  FoundryBuildStage *bundle;
  FoundryBuildStage *commit;
  FoundryBuildStage *dependencies;
  FoundryBuildStage *download;
  FoundryBuildStage *export;
  FoundryBuildStage *prepare;
  FoundryBuildStage *simple_build;
};

G_DEFINE_FINAL_TYPE (PluginFlatpakBuildAddin, plugin_flatpak_build_addin, FOUNDRY_TYPE_BUILD_ADDIN)

static void
ensure_documents_portal_cb (GObject      *object,
                            GAsyncResult *result,
                            gpointer      user_data)
{
  g_autoptr(GVariant) reply = NULL;
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (G_IS_DBUS_CONNECTION (object));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!(reply = g_dbus_connection_call_finish (G_DBUS_CONNECTION (object), result, &error)))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boolean (promise, TRUE);
}

static DexFuture *
ensure_documents_portal (void)
{
  g_autoptr(GError) error = NULL;
  g_autoptr(GDBusConnection) bus = g_bus_get_sync (G_BUS_TYPE_SESSION, NULL, &error);
  DexPromise *promise;

  if (bus == NULL)
    return dex_future_new_for_error (g_steal_pointer (&error));

  promise = dex_promise_new_cancellable ();
  g_dbus_connection_call (bus,
                          "org.freedesktop.portal.Documents",
                          "/org/freedesktop/portal/documents",
                          "org.freedesktop.portal.Documents",
                          "GetMountPoint",
                          g_variant_new ("()"),
                          NULL,
                          G_DBUS_CALL_FLAGS_NONE,
                          3000,
                          dex_promise_get_cancellable (promise),
                          ensure_documents_portal_cb,
                          dex_ref (promise));

  return DEX_FUTURE (promise);
}

static DexFuture *
plugin_flatpak_build_addin_load (FoundryBuildAddin *addin)
{
  PluginFlatpakBuildAddin *self = (PluginFlatpakBuildAddin *)addin;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundrySettings) settings = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autofree char *build_system = NULL;

  g_assert (PLUGIN_IS_FLATPAK_BUILD_ADDIN (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (addin));

  /* Ensure portal is setup */
  dex_await (ensure_documents_portal (), NULL);

  build_manager = foundry_context_dup_build_manager (context);
  pipeline = foundry_build_addin_dup_pipeline (addin);
  config = foundry_build_pipeline_dup_config (pipeline);
  build_system = foundry_context_dup_build_system (context);
  settings = foundry_context_load_settings (context, "app.devsuite.foundry.flatpak", NULL);

  g_signal_connect_object (settings,
                           "changed::state-dir",
                           G_CALLBACK (foundry_build_manager_invalidate),
                           build_manager,
                           G_CONNECT_SWAPPED);

  if (PLUGIN_IS_FLATPAK_CONFIG (config))
    {
      PluginFlatpakConfig *manifest = PLUGIN_FLATPAK_CONFIG (config);
      g_autoptr(GFile) file = plugin_flatpak_config_dup_file (manifest);
      g_autofree char *primary_module_name = plugin_flatpak_config_dup_primary_module_name (manifest);
      g_autofree char *manifest_path = g_file_get_path (file);
      g_autofree char *repo_dir = plugin_flatpak_get_repo_dir (context);
      g_autofree char *staging_dir = plugin_flatpak_get_staging_dir (pipeline);
      g_autofree char *state_dir = foundry_settings_get_string (settings, "state-dir");

      self->autogen = plugin_flatpak_autogen_stage_new (context, staging_dir);
      foundry_build_pipeline_add_stage (pipeline, self->autogen);

      self->prepare = plugin_flatpak_prepare_stage_new (context, repo_dir, staging_dir);
      foundry_build_pipeline_add_stage (pipeline, self->prepare);

      if (foundry_str_empty0 (state_dir))
        state_dir = foundry_context_cache_filename (context, "flatpak-builder", NULL);
      else
        foundry_path_expand_inplace (&state_dir);

      self->download = plugin_flatpak_download_stage_new (context, staging_dir, state_dir, manifest_path, primary_module_name);
      foundry_build_pipeline_add_stage (pipeline, self->download);

      self->dependencies = plugin_flatpak_dependencies_stage_new (context, staging_dir, state_dir, manifest_path, primary_module_name);
      foundry_build_pipeline_add_stage (pipeline, self->dependencies);

      if (foundry_str_equal0 (build_system, "flatpak-simple"))
        {
          g_autoptr(FoundryFlatpakModule) primary_module = plugin_flatpak_config_dup_primary_module (manifest);
          g_auto(GStrv) build_commands = foundry_flatpak_module_dup_build_commands (primary_module);

          self->simple_build = plugin_flatpak_simple_stage_new (context, (const char * const *)build_commands);
          foundry_build_pipeline_add_stage (pipeline, self->simple_build);
        }

      self->commit = plugin_flatpak_commit_stage_new (context, staging_dir, state_dir);
      foundry_build_pipeline_add_stage (pipeline, self->commit);

      self->export = plugin_flatpak_export_stage_new (context, staging_dir, state_dir, repo_dir);
      foundry_build_pipeline_add_stage (pipeline, self->export);

      self->bundle = plugin_flatpak_bundle_stage_new (context, staging_dir, state_dir, repo_dir);
      foundry_build_pipeline_add_stage (pipeline, self->bundle);
    }

  return dex_future_new_true ();
}

static DexFuture *
plugin_flatpak_build_addin_unload (FoundryBuildAddin *addin)
{
  PluginFlatpakBuildAddin *self = (PluginFlatpakBuildAddin *)addin;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;

  g_assert (PLUGIN_IS_FLATPAK_BUILD_ADDIN (self));

  pipeline = foundry_build_addin_dup_pipeline (addin);

  foundry_clear_build_stage (&self->autogen, pipeline);
  foundry_clear_build_stage (&self->download, pipeline);
  foundry_clear_build_stage (&self->dependencies, pipeline);
  foundry_clear_build_stage (&self->prepare, pipeline);
  foundry_clear_build_stage (&self->simple_build, pipeline);
  foundry_clear_build_stage (&self->commit, pipeline);
  foundry_clear_build_stage (&self->export, pipeline);
  foundry_clear_build_stage (&self->bundle, pipeline);

  return dex_future_new_true ();
}

static void
plugin_flatpak_build_addin_class_init (PluginFlatpakBuildAddinClass *klass)
{
  FoundryBuildAddinClass *build_addin_class = FOUNDRY_BUILD_ADDIN_CLASS (klass);

  build_addin_class->load = plugin_flatpak_build_addin_load;
  build_addin_class->unload = plugin_flatpak_build_addin_unload;
}

static void
plugin_flatpak_build_addin_init (PluginFlatpakBuildAddin *self)
{
}
