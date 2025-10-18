/* plugin-flatpak-autogen-stage.c
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

#include "plugin-flatpak.h"
#include "plugin-flatpak-autogen-stage.h"
#include "plugin-flatpak-config.h"

struct _PluginFlatpakAutogenStage
{
  FoundryBuildStage parent_instance;
  char *staging_dir;
};

G_DEFINE_FINAL_TYPE (PluginFlatpakAutogenStage, plugin_flatpak_autogen_stage, FOUNDRY_TYPE_BUILD_STAGE)

enum {
  PROP_0,
  PROP_STAGING_DIR,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static DexFuture *
plugin_flatpak_autogen_stage_build_fiber (gpointer data)
{
  FoundryPair *state = data;
  g_autoptr(PluginFlatpakAutogenStage) self = NULL;
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *arch = NULL;
  g_autofree char *app_id = NULL;
  g_autofree char *runtime = NULL;
  g_autofree char *runtime_version = NULL;
  g_autofree char *sdk = NULL;

  g_assert (state != NULL);
  g_assert (PLUGIN_IS_FLATPAK_AUTOGEN_STAGE (state->first));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (state->second));

  self = g_object_ref (PLUGIN_FLATPAK_AUTOGEN_STAGE (state->first));
  progress = g_object_ref (FOUNDRY_BUILD_PROGRESS (state->second));
  pipeline = foundry_build_stage_dup_pipeline (FOUNDRY_BUILD_STAGE (self));
  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (pipeline));
  cancellable = foundry_build_progress_dup_cancellable (progress);

  dex_return_error_if_fail (FOUNDRY_IS_BUILD_PIPELINE (pipeline));

  arch = foundry_build_pipeline_dup_arch (pipeline);
  config = foundry_build_pipeline_dup_config (pipeline);

  dex_return_error_if_fail (arch != NULL);
  dex_return_error_if_fail (PLUGIN_IS_FLATPAK_CONFIG (config));

  app_id = plugin_flatpak_config_dup_id (PLUGIN_FLATPAK_CONFIG (config));
  sdk = plugin_flatpak_config_dup_sdk (PLUGIN_FLATPAK_CONFIG (config));
  runtime = plugin_flatpak_config_dup_runtime (PLUGIN_FLATPAK_CONFIG (config));
  runtime_version = plugin_flatpak_config_dup_runtime_version (PLUGIN_FLATPAK_CONFIG (config));

  if (runtime == NULL && sdk != NULL)
    g_set_str (&runtime, sdk);
  else if (sdk == NULL && runtime != NULL)
    g_set_str (&sdk, runtime);

  dex_return_error_if_fail (app_id != NULL);
  dex_return_error_if_fail (sdk != NULL);
  dex_return_error_if_fail (runtime != NULL);
  dex_return_error_if_fail (runtime_version != NULL);

  foundry_build_progress_print (progress, "%s\n", _("Running flatpak build-init"));

  launcher = foundry_process_launcher_new ();

  if (!dex_await (foundry_build_pipeline_prepare (pipeline, launcher, FOUNDRY_BUILD_PIPELINE_PHASE_AUTOGEN), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  foundry_process_launcher_append_argv (launcher, "flatpak");
  foundry_process_launcher_append_argv (launcher, "build-init");
  foundry_process_launcher_append_argv (launcher, "--type=app");
  foundry_process_launcher_append_formatted (launcher, "--arch=%s", arch);
  foundry_process_launcher_append_argv (launcher, self->staging_dir);
  foundry_process_launcher_append_argv (launcher, app_id);
  foundry_process_launcher_append_argv (launcher, sdk);
  foundry_process_launcher_append_argv (launcher, runtime);
  foundry_process_launcher_append_argv (launcher, runtime_version);

  /* TODO: --base=APP --base-version= --base-extension= */

  foundry_build_progress_setup_pty (progress, launcher);

  plugin_flatpak_apply_config_dir (context, launcher);

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return foundry_subprocess_wait_check (subprocess, cancellable);
}

static DexFuture *
plugin_flatpak_autogen_stage_build (FoundryBuildStage    *stage,
                                    FoundryBuildProgress *progress)
{
  g_assert (PLUGIN_IS_FLATPAK_AUTOGEN_STAGE (stage));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_flatpak_autogen_stage_build_fiber,
                              foundry_pair_new (stage, progress),
                              (GDestroyNotify) foundry_pair_free);
}

static DexFuture *
plugin_flatpak_autogen_stage_query_fiber (gpointer data)
{
  PluginFlatpakAutogenStage *self = data;
  g_autofree char *metadata = NULL;
  g_autofree char *var = NULL;
  g_autofree char *files = NULL;
  gboolean completed = FALSE;

  g_assert (PLUGIN_IS_FLATPAK_AUTOGEN_STAGE (self));

  files = g_build_filename (self->staging_dir, "files", NULL);
  var = g_build_filename (self->staging_dir, "var", NULL);
  metadata = g_build_filename (self->staging_dir, "metadata", NULL);

  if (dex_await_boolean (foundry_file_test (self->staging_dir, G_FILE_TEST_IS_DIR), NULL) &&
      dex_await_boolean (foundry_file_test (files, G_FILE_TEST_IS_DIR), NULL) &&
      dex_await_boolean (foundry_file_test (var, G_FILE_TEST_IS_DIR), NULL) &&
      dex_await_boolean (foundry_file_test (metadata, G_FILE_TEST_IS_REGULAR), NULL))
    completed = TRUE;

  foundry_build_stage_set_completed (FOUNDRY_BUILD_STAGE (self), completed);

  if (!completed)
    {
      g_autoptr(FoundryDirectoryReaper) reaper = NULL;
      g_autoptr(GFile) staging_dir = NULL;

      FOUNDRY_CONTEXTUAL_MESSAGE (self, "%s", _("Removing stale flatpak staging directory"));

      staging_dir = g_file_new_for_path (self->staging_dir);

      reaper = foundry_directory_reaper_new ();
      foundry_directory_reaper_add_directory (reaper, staging_dir, 0);
      foundry_directory_reaper_add_file (reaper, staging_dir, 0);

      dex_await (foundry_directory_reaper_execute (reaper), NULL);
    }

  return dex_future_new_true ();
}

static DexFuture *
plugin_flatpak_autogen_stage_query (FoundryBuildStage *stage)
{
  g_assert (PLUGIN_IS_FLATPAK_AUTOGEN_STAGE (stage));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_flatpak_autogen_stage_query_fiber,
                              g_object_ref (stage),
                              g_object_unref);
}

static DexFuture *
plugin_flatpak_autogen_stage_purge (FoundryBuildStage    *stage,
                                    FoundryBuildProgress *progress)
{
  PluginFlatpakAutogenStage *self = (PluginFlatpakAutogenStage *)stage;
  g_autoptr(FoundryDirectoryReaper) reaper = NULL;
  g_autoptr(GFile) staging_dir = NULL;

  g_assert (PLUGIN_IS_FLATPAK_AUTOGEN_STAGE (self));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));

  staging_dir = g_file_new_for_path (self->staging_dir);

  reaper = foundry_directory_reaper_new ();
  foundry_directory_reaper_add_directory (reaper, staging_dir, 0);
  foundry_directory_reaper_add_file (reaper, staging_dir, 0);

  return foundry_directory_reaper_execute (reaper);
}

static FoundryBuildPipelinePhase
plugin_flatpak_autogen_stage_get_phase (FoundryBuildStage *stage)
{
  return FOUNDRY_BUILD_PIPELINE_PHASE_AUTOGEN | FOUNDRY_BUILD_PIPELINE_PHASE_BEFORE;
}

static void
plugin_flatpak_autogen_stage_finalize (GObject *object)
{
  PluginFlatpakAutogenStage *self = (PluginFlatpakAutogenStage *)object;

  g_clear_pointer (&self->staging_dir, g_free);

  G_OBJECT_CLASS (plugin_flatpak_autogen_stage_parent_class)->finalize (object);
}

static void
plugin_flatpak_autogen_stage_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  PluginFlatpakAutogenStage *self = PLUGIN_FLATPAK_AUTOGEN_STAGE (object);

  switch (prop_id)
    {
    case PROP_STAGING_DIR:
      g_value_set_string (value, self->staging_dir);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_autogen_stage_set_property (GObject      *object,
                                           guint         prop_id,
                                           const GValue *value,
                                           GParamSpec   *pspec)
{
  PluginFlatpakAutogenStage *self = PLUGIN_FLATPAK_AUTOGEN_STAGE (object);

  switch (prop_id)
    {
    case PROP_STAGING_DIR:
      self->staging_dir = g_value_dup_string (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_autogen_stage_class_init (PluginFlatpakAutogenStageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryBuildStageClass *build_stage_class = FOUNDRY_BUILD_STAGE_CLASS (klass);

  object_class->finalize = plugin_flatpak_autogen_stage_finalize;
  object_class->get_property = plugin_flatpak_autogen_stage_get_property;
  object_class->set_property = plugin_flatpak_autogen_stage_set_property;

  build_stage_class->get_phase = plugin_flatpak_autogen_stage_get_phase;
  build_stage_class->build = plugin_flatpak_autogen_stage_build;
  build_stage_class->query = plugin_flatpak_autogen_stage_query;
  build_stage_class->purge = plugin_flatpak_autogen_stage_purge;

  properties[PROP_STAGING_DIR] =
    g_param_spec_string ("staging-dir", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_flatpak_autogen_stage_init (PluginFlatpakAutogenStage *self)
{
}

FoundryBuildStage *
plugin_flatpak_autogen_stage_new (FoundryContext *context,
                                  const char     *staging_dir)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (staging_dir != NULL, NULL);

  return g_object_new (PLUGIN_TYPE_FLATPAK_AUTOGEN_STAGE,
                       "kind", "flatpak",
                       "title", _("Initialize from Flatpak Manifest"),
                       "context", context,
                       "staging-dir", staging_dir,
                       NULL);
}
