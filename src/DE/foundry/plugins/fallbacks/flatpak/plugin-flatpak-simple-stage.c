/* plugin-flatpak-simple-stage.c
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

#include "plugin-flatpak-simple-stage.h"

struct _PluginFlatpakSimpleStage
{
  FoundryBuildStage parent_instance;
  char **commands;
};

enum {
  PROP_0,
  PROP_COMMANDS,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (PluginFlatpakSimpleStage, plugin_flatpak_simple_stage, FOUNDRY_TYPE_BUILD_STAGE)

static GParamSpec *properties[N_PROPS];

static DexFuture *
plugin_flatpak_simple_stage_build_fiber (gpointer data)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  PluginFlatpakSimpleStage *self;
  FoundryBuildProgress *progress;
  FoundryPair *pair = data;

  g_assert (pair != NULL);
  g_assert (PLUGIN_IS_FLATPAK_SIMPLE_STAGE (pair->first));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (pair->second));

  self = PLUGIN_FLATPAK_SIMPLE_STAGE (pair->first);
  progress = FOUNDRY_BUILD_PROGRESS (pair->second);
  pipeline = foundry_build_stage_dup_pipeline (FOUNDRY_BUILD_STAGE (self));
  cancellable = foundry_build_progress_dup_cancellable (progress);

  for (guint i = 0; self->commands[i]; i++)
    {
      g_autoptr(FoundryProcessLauncher) launcher = NULL;
      g_autoptr(GSubprocess) subprocess = NULL;
      g_autoptr(GError) error = NULL;
      g_auto(GStrv) argv = NULL;
      int argc;

      /* Make sure we can parse the command argv */
      if (!g_shell_parse_argv (self->commands[i], &argc, &argv, &error))
        return dex_future_new_for_error (g_steal_pointer (&error));

      launcher = foundry_process_launcher_new ();

      /* Prepare the pipeline for execution */
      if (!dex_await (foundry_build_pipeline_prepare (pipeline, launcher, FOUNDRY_BUILD_PIPELINE_PHASE_BUILD), &error))
        return dex_future_new_for_error (g_steal_pointer (&error));

      /* Make sure variables are expanded from environment */
      foundry_process_launcher_push_shell (launcher, 0);

      /* Setup argv for subprocess */
      foundry_process_launcher_append_args (launcher, (const char * const *)argv);

      /* Setup the PTY so output ends up in the right place */
      foundry_build_progress_setup_pty (progress, launcher);

      /* Spawn the process within the pipeline */
      if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));

      /* Await completion of subprocess but possibly force-exit it on cancellation */
      if (!dex_await (foundry_subprocess_wait_check (subprocess, cancellable), &error))
        return dex_future_new_for_error (g_steal_pointer (&error));
    }

  return dex_future_new_true ();
}

static DexFuture *
plugin_flatpak_simple_stage_build (FoundryBuildStage    *build_stage,
                                   FoundryBuildProgress *progress)
{
  PluginFlatpakSimpleStage *self = (PluginFlatpakSimpleStage *)build_stage;

  g_assert (PLUGIN_IS_FLATPAK_SIMPLE_STAGE (build_stage));
  g_assert (FOUNDRY_IS_BUILD_PROGRESS (progress));

  if (self->commands == NULL || self->commands[0] == NULL)
    return dex_future_new_true ();

  return dex_scheduler_spawn (NULL, 0,
                              plugin_flatpak_simple_stage_build_fiber,
                              foundry_pair_new (build_stage, progress),
                              (GDestroyNotify) foundry_pair_free);
}

static FoundryBuildPipelinePhase
plugin_flatpak_simple_stage_get_phase (FoundryBuildStage *build_stage)
{
  return FOUNDRY_BUILD_PIPELINE_PHASE_BUILD;
}

static void
plugin_flatpak_simple_stage_finalize (GObject *object)
{
  PluginFlatpakSimpleStage *self = (PluginFlatpakSimpleStage *)object;

  g_clear_pointer (&self->commands, g_strfreev);

  G_OBJECT_CLASS (plugin_flatpak_simple_stage_parent_class)->finalize (object);
}

static void
plugin_flatpak_simple_stage_get_property (GObject    *object,
                                          guint       prop_id,
                                          GValue     *value,
                                          GParamSpec *pspec)
{
  PluginFlatpakSimpleStage *self = PLUGIN_FLATPAK_SIMPLE_STAGE (object);

  switch (prop_id)
    {
    case PROP_COMMANDS:
      g_value_set_boxed (value, self->commands);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_simple_stage_set_property (GObject      *object,
                                          guint         prop_id,
                                          const GValue *value,
                                          GParamSpec   *pspec)
{
  PluginFlatpakSimpleStage *self = PLUGIN_FLATPAK_SIMPLE_STAGE (object);

  switch (prop_id)
    {
    case PROP_COMMANDS:
      self->commands = g_value_dup_boxed (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_simple_stage_class_init (PluginFlatpakSimpleStageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryBuildStageClass *build_stage_class = FOUNDRY_BUILD_STAGE_CLASS (klass);

  object_class->finalize = plugin_flatpak_simple_stage_finalize;
  object_class->get_property = plugin_flatpak_simple_stage_get_property;
  object_class->set_property = plugin_flatpak_simple_stage_set_property;

  build_stage_class->get_phase = plugin_flatpak_simple_stage_get_phase;
  build_stage_class->build = plugin_flatpak_simple_stage_build;

  properties[PROP_COMMANDS] =
    g_param_spec_boxed ("commands", NULL, NULL,
                        G_TYPE_STRV,
                        (G_PARAM_READWRITE |
                         G_PARAM_EXPLICIT_NOTIFY |
                         G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_flatpak_simple_stage_init (PluginFlatpakSimpleStage *self)
{
}

FoundryBuildStage *
plugin_flatpak_simple_stage_new (FoundryContext     *context,
                                 const char * const *commands)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);

  return g_object_new (PLUGIN_TYPE_FLATPAK_SIMPLE_STAGE,
                       "context", context,
                       "commands", commands,
                       "kind", "flatpak",
                       "title", _("Build Project"),
                       NULL);
}
