/* plugin-cmake-prepare-stage.c
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

#include "plugin-cmake-prepare-stage.h"

struct _PluginCmakePrepareStage
{
  FoundryBuildStage parent_instance;
  char *builddir;
};

G_DEFINE_FINAL_TYPE (PluginCmakePrepareStage, plugin_cmake_prepare_stage, FOUNDRY_TYPE_BUILD_STAGE)

static GBytes *query_bytes;

static FoundryBuildPipelinePhase
plugin_cmake_prepare_stage_get_phase (FoundryBuildStage *build_stage)
{
  return FOUNDRY_BUILD_PIPELINE_PHASE_PREPARE;
}

static DexFuture *
plugin_cmake_prepare_stage_build_fiber (gpointer data)
{
  PluginCmakePrepareStage *self = data;
  g_autofree char *query_dir = NULL;
  g_autoptr(GFile) query_file = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (PLUGIN_IS_CMAKE_PREPARE_STAGE (self));

  query_dir = g_build_filename (self->builddir, ".cmake", "api", "v1", "query", "client-builder", NULL);
  query_file = g_file_new_build_filename (query_dir, "query.json", NULL);

  if (!dex_await (dex_mkdir_with_parents (query_dir, 0750), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!dex_await (dex_file_replace_contents_bytes (query_file, query_bytes, NULL, FALSE, G_FILE_CREATE_REPLACE_DESTINATION), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static DexFuture *
plugin_cmake_prepare_stage_build (FoundryBuildStage    *build_stage,
                                  FoundryBuildProgress *progress)
{
  return dex_scheduler_spawn (NULL, 0,
                              plugin_cmake_prepare_stage_build_fiber,
                              g_object_ref (build_stage),
                              g_object_unref);
}

static void
plugin_cmake_prepare_stage_finalize (GObject *object)
{
  PluginCmakePrepareStage *self = (PluginCmakePrepareStage *)object;

  g_clear_pointer (&self->builddir, g_free);

  G_OBJECT_CLASS (plugin_cmake_prepare_stage_parent_class)->finalize (object);
}

static void
plugin_cmake_prepare_stage_class_init (PluginCmakePrepareStageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryBuildStageClass *build_stage_class = FOUNDRY_BUILD_STAGE_CLASS (klass);
  const char *query_string = "{\"requests\":[{\"kind\":\"codemodel\",\"version\":2}]}";

  object_class->finalize = plugin_cmake_prepare_stage_finalize;

  build_stage_class->get_phase = plugin_cmake_prepare_stage_get_phase;
  build_stage_class->build = plugin_cmake_prepare_stage_build;

  query_bytes = g_bytes_new_static (query_string, strlen (query_string));
}

static void
plugin_cmake_prepare_stage_init (PluginCmakePrepareStage *self)
{
  foundry_build_stage_set_kind (FOUNDRY_BUILD_STAGE (self), "cmake");
  foundry_build_stage_set_title (FOUNDRY_BUILD_STAGE (self), _("Prepare Build Environment"));
}

FoundryBuildStage *
plugin_cmake_prepare_stage_new (FoundryContext *context,
                                const char     *builddir)
{
  PluginCmakePrepareStage *self;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (builddir != NULL, NULL);

  self = g_object_new (PLUGIN_TYPE_CMAKE_PREPARE_STAGE,
                       "context", context,
                       NULL);
  self->builddir = g_strdup (builddir);

  return FOUNDRY_BUILD_STAGE (self);
}
