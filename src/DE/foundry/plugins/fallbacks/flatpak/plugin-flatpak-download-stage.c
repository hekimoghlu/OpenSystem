/* plugin-flatpak-download-stage.c
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

#include "plugin-flatpak.h"
#include "plugin-flatpak-download-stage.h"

#include "foundry-util-private.h"

struct _PluginFlatpakDownloadStage
{
  FoundryBuildStage parent_instance;
  char *manifest_path;
  char *staging_dir;
  char *state_dir;
  char *primary_module_name;
};

G_DEFINE_FINAL_TYPE (PluginFlatpakDownloadStage, plugin_flatpak_download_stage, FOUNDRY_TYPE_BUILD_STAGE)

enum {
  PROP_0,
  PROP_MANIFEST_PATH,
  PROP_PRIMARY_MODULE_NAME,
  PROP_STATE_DIR,
  PROP_STAGING_DIR,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static DexFuture *
plugin_flatpak_download_stage_build_fiber (gpointer user_data)
{
  FoundryPair *pair = user_data;
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  PluginFlatpakDownloadStage *self;
  FoundryBuildProgress *progress;
  g_autofree char *arch = NULL;

  g_assert (pair != NULL);
  g_assert (PLUGIN_IS_FLATPAK_DOWNLOAD_STAGE (pair->first));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (pair->second));

  self = PLUGIN_FLATPAK_DOWNLOAD_STAGE (pair->first);
  progress = FOUNDRY_BUILD_PROGRESS (pair->second);
  cancellable = foundry_build_progress_dup_cancellable (progress);
  pipeline = foundry_build_stage_dup_pipeline (FOUNDRY_BUILD_STAGE (self));
  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  launcher = foundry_process_launcher_new ();
  arch = foundry_build_pipeline_dup_arch (pipeline);

  foundry_process_launcher_append_argv (launcher, "flatpak-builder");
  foundry_process_launcher_append_formatted (launcher, "--arch=%s", arch);
  foundry_process_launcher_append_argv (launcher, "--ccache");
  foundry_process_launcher_append_argv (launcher, "--force-clean");
  foundry_process_launcher_append_argv (launcher, "--state-dir");
  foundry_process_launcher_append_argv (launcher, self->state_dir);
  foundry_process_launcher_append_argv (launcher, "--download-only");
  foundry_process_launcher_append_argv (launcher, "--disable-updates");

  if (self->primary_module_name != NULL)
    foundry_process_launcher_append_formatted (launcher, "--stop-at=%s", self->primary_module_name);

  foundry_process_launcher_append_argv (launcher, self->staging_dir);
  foundry_process_launcher_append_argv (launcher, self->manifest_path);

  foundry_build_progress_setup_pty (progress, launcher);

  plugin_flatpak_apply_config_dir (context, launcher);

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return foundry_subprocess_wait_check (subprocess, cancellable);
}

static DexFuture *
plugin_flatpak_download_stage_build (FoundryBuildStage    *build_stage,
                                     FoundryBuildProgress *progress)
{
  PluginFlatpakDownloadStage *self = (PluginFlatpakDownloadStage *)build_stage;
  g_autoptr(FoundryContext) context = NULL;

  g_assert (PLUGIN_IS_FLATPAK_DOWNLOAD_STAGE (self));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  if (!foundry_context_network_allowed (context))
    return dex_future_new_true ();

  return dex_scheduler_spawn (NULL, 0,
                              plugin_flatpak_download_stage_build_fiber,
                              foundry_pair_new (build_stage, progress),
                              (GDestroyNotify) foundry_pair_free);
}

static FoundryBuildPipelinePhase
plugin_flatpak_download_stage_get_phase (FoundryBuildStage *build_stage)
{
  return FOUNDRY_BUILD_PIPELINE_PHASE_DOWNLOADS;
}

static void
plugin_flatpak_download_stage_finalize (GObject *object)
{
  PluginFlatpakDownloadStage *self = (PluginFlatpakDownloadStage *)object;

  g_clear_pointer (&self->manifest_path, g_free);
  g_clear_pointer (&self->primary_module_name, g_free);
  g_clear_pointer (&self->staging_dir, g_free);
  g_clear_pointer (&self->state_dir, g_free);

  G_OBJECT_CLASS (plugin_flatpak_download_stage_parent_class)->finalize (object);
}

static void
plugin_flatpak_download_stage_get_property (GObject    *object,
                                            guint       prop_id,
                                            GValue     *value,
                                            GParamSpec *pspec)
{
  PluginFlatpakDownloadStage *self = PLUGIN_FLATPAK_DOWNLOAD_STAGE (object);

  switch (prop_id)
    {
    case PROP_MANIFEST_PATH:
      g_value_set_string (value, self->manifest_path);
      break;

    case PROP_PRIMARY_MODULE_NAME:
      g_value_set_string (value, self->primary_module_name);
      break;

    case PROP_STAGING_DIR:
      g_value_set_string (value, self->staging_dir);
      break;

    case PROP_STATE_DIR:
      g_value_set_string (value, self->state_dir);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_download_stage_set_property (GObject      *object,
                                            guint         prop_id,
                                            const GValue *value,
                                            GParamSpec   *pspec)
{
  PluginFlatpakDownloadStage *self = PLUGIN_FLATPAK_DOWNLOAD_STAGE (object);

  switch (prop_id)
    {
    case PROP_MANIFEST_PATH:
      self->manifest_path = g_value_dup_string (value);
      break;

    case PROP_PRIMARY_MODULE_NAME:
      self->primary_module_name = g_value_dup_string (value);
      break;

    case PROP_STAGING_DIR:
      self->staging_dir = g_value_dup_string (value);
      break;

    case PROP_STATE_DIR:
      self->state_dir = g_value_dup_string (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_download_stage_class_init (PluginFlatpakDownloadStageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryBuildStageClass *build_stage_class = FOUNDRY_BUILD_STAGE_CLASS (klass);

  object_class->finalize = plugin_flatpak_download_stage_finalize;
  object_class->get_property = plugin_flatpak_download_stage_get_property;
  object_class->set_property = plugin_flatpak_download_stage_set_property;

  build_stage_class->build = plugin_flatpak_download_stage_build;
  build_stage_class->get_phase = plugin_flatpak_download_stage_get_phase;

  properties[PROP_MANIFEST_PATH] =
    g_param_spec_string ("manifest-path", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PRIMARY_MODULE_NAME] =
    g_param_spec_string ("primary-module-name", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_STAGING_DIR] =
    g_param_spec_string ("staging-dir", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_STATE_DIR] =
    g_param_spec_string ("state-dir", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_flatpak_download_stage_init (PluginFlatpakDownloadStage *self)
{
}

FoundryBuildStage *
plugin_flatpak_download_stage_new (FoundryContext *context,
                                   const char     *staging_dir,
                                   const char     *state_dir,
                                   const char     *manifest_path,
                                   const char     *primary_module_name)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (state_dir != NULL, NULL);
  g_return_val_if_fail (staging_dir != NULL, NULL);
  g_return_val_if_fail (manifest_path != NULL, NULL);

  return g_object_new (PLUGIN_TYPE_FLATPAK_DOWNLOAD_STAGE,
                       "context", context,
                       "manifest-path", manifest_path,
                       "primary-module-name", primary_module_name,
                       "staging-dir", staging_dir,
                       "state-dir", state_dir,
                       "kind", "flatpak",
                       "title", _("Download Flatpak Dependencies"),
                       NULL);
}
