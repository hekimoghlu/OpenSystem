/* plugin-flatpak-commit-stage.c
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
#include "plugin-flatpak-config.h"
#include "plugin-flatpak-commit-stage.h"

#include "foundry-util-private.h"

struct _PluginFlatpakCommitStage
{
  FoundryBuildStage parent_instance;
  char *staging_dir;
  char *state_dir;
};

G_DEFINE_FINAL_TYPE (PluginFlatpakCommitStage, plugin_flatpak_commit_stage, FOUNDRY_TYPE_BUILD_STAGE)

enum {
  PROP_0,
  PROP_STATE_DIR,
  PROP_STAGING_DIR,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static DexFuture *
plugin_flatpak_commit_stage_build_fiber (gpointer user_data)
{
  FoundryPair *pair = user_data;
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryFlatpakManifest) manifest = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  g_auto(GStrv) finish_args = NULL;
  g_autofree char *command = NULL;
  PluginFlatpakCommitStage *self;
  FoundryBuildProgress *progress;

  g_assert (pair != NULL);
  g_assert (PLUGIN_IS_FLATPAK_COMMIT_STAGE (pair->first));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (pair->second));

  self = PLUGIN_FLATPAK_COMMIT_STAGE (pair->first);
  progress = FOUNDRY_BUILD_PROGRESS (pair->second);
  cancellable = foundry_build_progress_dup_cancellable (progress);
  pipeline = foundry_build_stage_dup_pipeline (FOUNDRY_BUILD_STAGE (self));
  config = foundry_build_pipeline_dup_config (pipeline);
  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  launcher = foundry_process_launcher_new ();

  if (!PLUGIN_IS_FLATPAK_CONFIG (config))
    return dex_future_new_true ();

  manifest = plugin_flatpak_config_dup_manifest (PLUGIN_FLATPAK_CONFIG (config));

  foundry_process_launcher_push_host (launcher);

  foundry_process_launcher_append_argv (launcher, "flatpak");
  foundry_process_launcher_append_argv (launcher, "build-finish");

  command = foundry_flatpak_manifest_dup_command (manifest);
  foundry_process_launcher_append_formatted (launcher, "--command=%s", command);

  if ((finish_args = foundry_flatpak_manifest_dup_finish_args (manifest)))
    foundry_process_launcher_append_args (launcher,
                                          (const char * const *)finish_args);

  foundry_process_launcher_append_argv (launcher, self->staging_dir);

  foundry_build_progress_setup_pty (progress, launcher);

  plugin_flatpak_apply_config_dir (context, launcher);

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return foundry_subprocess_wait_check (subprocess, cancellable);
}

static DexFuture *
plugin_flatpak_commit_stage_build (FoundryBuildStage    *build_stage,
                                   FoundryBuildProgress *progress)
{
  PluginFlatpakCommitStage *self = (PluginFlatpakCommitStage *)build_stage;

  g_assert (PLUGIN_IS_FLATPAK_COMMIT_STAGE (self));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_flatpak_commit_stage_build_fiber,
                              foundry_pair_new (build_stage, progress),
                              (GDestroyNotify) foundry_pair_free);
}

static FoundryBuildPipelinePhase
plugin_flatpak_commit_stage_get_phase (FoundryBuildStage *build_stage)
{
  return FOUNDRY_BUILD_PIPELINE_PHASE_COMMIT;
}

static void
plugin_flatpak_commit_stage_finalize (GObject *object)
{
  PluginFlatpakCommitStage *self = (PluginFlatpakCommitStage *)object;

  g_clear_pointer (&self->staging_dir, g_free);
  g_clear_pointer (&self->state_dir, g_free);

  G_OBJECT_CLASS (plugin_flatpak_commit_stage_parent_class)->finalize (object);
}

static void
plugin_flatpak_commit_stage_get_property (GObject    *object,
                                          guint       prop_id,
                                          GValue     *value,
                                          GParamSpec *pspec)
{
  PluginFlatpakCommitStage *self = PLUGIN_FLATPAK_COMMIT_STAGE (object);

  switch (prop_id)
    {
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
plugin_flatpak_commit_stage_set_property (GObject      *object,
                                          guint         prop_id,
                                          const GValue *value,
                                          GParamSpec   *pspec)
{
  PluginFlatpakCommitStage *self = PLUGIN_FLATPAK_COMMIT_STAGE (object);

  switch (prop_id)
    {
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
plugin_flatpak_commit_stage_class_init (PluginFlatpakCommitStageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryBuildStageClass *build_stage_class = FOUNDRY_BUILD_STAGE_CLASS (klass);

  object_class->finalize = plugin_flatpak_commit_stage_finalize;
  object_class->get_property = plugin_flatpak_commit_stage_get_property;
  object_class->set_property = plugin_flatpak_commit_stage_set_property;

  build_stage_class->build = plugin_flatpak_commit_stage_build;
  build_stage_class->get_phase = plugin_flatpak_commit_stage_get_phase;

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
plugin_flatpak_commit_stage_init (PluginFlatpakCommitStage *self)
{
}

FoundryBuildStage *
plugin_flatpak_commit_stage_new (FoundryContext *context,
                                 const char     *staging_dir,
                                 const char     *state_dir)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (state_dir != NULL, NULL);
  g_return_val_if_fail (staging_dir != NULL, NULL);

  return g_object_new (PLUGIN_TYPE_FLATPAK_COMMIT_STAGE,
                       "context", context,
                       "staging-dir", staging_dir,
                       "state-dir", state_dir,
                       "kind", "flatpak",
                       "title", _("Apply Finishing Details"),
                       NULL);
}
