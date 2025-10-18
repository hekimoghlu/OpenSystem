/* plugin-flatpak-prepare-stage.c
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

#include "plugin-flatpak-prepare-stage.h"

struct _PluginFlatpakPrepareStage
{
  FoundryBuildStage parent_instance;
  char *repo_dir;
  char *staging_dir;
};

enum {
  PROP_0,
  PROP_REPO_DIR,
  PROP_STAGING_DIR,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (PluginFlatpakPrepareStage, plugin_flatpak_prepare_stage, FOUNDRY_TYPE_BUILD_STAGE)

static GParamSpec *properties[N_PROPS];

static DexFuture *
plugin_flatpak_prepare_stage_query (FoundryBuildStage *stage)
{
  PluginFlatpakPrepareStage *self = (PluginFlatpakPrepareStage *)stage;
  gboolean completed = FALSE;

  g_assert (PLUGIN_IS_FLATPAK_PREPARE_STAGE (self));

  if (dex_await_boolean (foundry_file_test (self->repo_dir, G_FILE_TEST_IS_DIR), NULL) &&
      dex_await_boolean (foundry_file_test (self->staging_dir, G_FILE_TEST_IS_DIR), NULL))
    completed = TRUE;

  foundry_build_stage_set_completed (stage, completed);

  return dex_future_new_true ();
}

static DexFuture *
plugin_flatpak_prepare_stage_build (FoundryBuildStage    *stage,
                                    FoundryBuildProgress *progress)
{
  PluginFlatpakPrepareStage *self = (PluginFlatpakPrepareStage *)stage;
  g_autoptr(GError) error = NULL;

  g_assert (PLUGIN_IS_FLATPAK_PREPARE_STAGE (self));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));

  foundry_build_progress_print (progress, "%s\n", _("Creating Flatpak staging directories"));

  if (!dex_await (dex_mkdir_with_parents (self->repo_dir, 0750), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!dex_await (dex_mkdir_with_parents (self->staging_dir, 0750), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static FoundryBuildPipelinePhase
plugin_flatpak_prepare_stage_get_phase (FoundryBuildStage *stage)
{
  return FOUNDRY_BUILD_PIPELINE_PHASE_PREPARE;
}

static void
plugin_flatpak_prepare_stage_finalize (GObject *object)
{
  PluginFlatpakPrepareStage *self = (PluginFlatpakPrepareStage *)object;

  g_clear_pointer (&self->repo_dir, g_free);
  g_clear_pointer (&self->staging_dir, g_free);

  G_OBJECT_CLASS (plugin_flatpak_prepare_stage_parent_class)->finalize (object);
}

static void
plugin_flatpak_prepare_stage_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  PluginFlatpakPrepareStage *self = PLUGIN_FLATPAK_PREPARE_STAGE (object);

  switch (prop_id)
    {
    case PROP_REPO_DIR:
      g_value_set_string (value, self->repo_dir);
      break;

    case PROP_STAGING_DIR:
      g_value_set_string (value, self->staging_dir);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_prepare_stage_set_property (GObject      *object,
                                           guint         prop_id,
                                           const GValue *value,
                                           GParamSpec   *pspec)
{
  PluginFlatpakPrepareStage *self = PLUGIN_FLATPAK_PREPARE_STAGE (object);

  switch (prop_id)
    {
    case PROP_REPO_DIR:
      self->repo_dir = g_value_dup_string (value);
      break;

    case PROP_STAGING_DIR:
      self->staging_dir = g_value_dup_string (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_prepare_stage_class_init (PluginFlatpakPrepareStageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryBuildStageClass *build_stage_class = FOUNDRY_BUILD_STAGE_CLASS (klass);

  object_class->finalize = plugin_flatpak_prepare_stage_finalize;
  object_class->get_property = plugin_flatpak_prepare_stage_get_property;
  object_class->set_property = plugin_flatpak_prepare_stage_set_property;

  build_stage_class->get_phase = plugin_flatpak_prepare_stage_get_phase;
  build_stage_class->build = plugin_flatpak_prepare_stage_build;
  build_stage_class->query = plugin_flatpak_prepare_stage_query;

  properties[PROP_REPO_DIR] =
    g_param_spec_string ("repo-dir", NULL, NULL,
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

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_flatpak_prepare_stage_init (PluginFlatpakPrepareStage *self)
{
}

FoundryBuildStage *
plugin_flatpak_prepare_stage_new (FoundryContext *context,
                                  const char     *repo_dir,
                                  const char     *staging_dir)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (repo_dir != NULL, NULL);
  g_return_val_if_fail (staging_dir != NULL, NULL);

  return g_object_new (PLUGIN_TYPE_FLATPAK_PREPARE_STAGE,
                       "kind", "flatpak",
                       "title", _("Prepare Staging Directories"),
                       "context", context,
                       "repo-dir", repo_dir,
                       "staging-dir", staging_dir,
                       NULL);
}
