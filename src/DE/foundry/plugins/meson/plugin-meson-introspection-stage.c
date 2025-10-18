/* plugin-meson-introspection-stage.c
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

#include "plugin-meson-build-target.h"
#include "plugin-meson-introspection-stage.h"
#include "plugin-meson-test.h"

struct _PluginMesonIntrospectionStage
{
  PluginMesonBaseStage  parent_instance;
  JsonNode             *introspection;
};

G_DEFINE_FINAL_TYPE (PluginMesonIntrospectionStage, plugin_meson_introspection_stage, PLUGIN_TYPE_MESON_BASE_STAGE)

static FoundryBuildPipelinePhase
plugin_meson_introspection_stage_get_phase (FoundryBuildStage *stage)
{
  return FOUNDRY_BUILD_PIPELINE_PHASE_CONFIGURE | FOUNDRY_BUILD_PIPELINE_PHASE_AFTER;
}

static DexFuture *
plugin_meson_introspection_stage_build_fiber (FoundryBuildStage    *stage,
                                              FoundryBuildProgress *progress)
{
  PluginMesonIntrospectionStage *self = (PluginMesonIntrospectionStage *)stage;
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(JsonParser) parser = NULL;
  g_autoptr(GIOStream) io_stream = NULL;
  g_autoptr(GError) error = NULL;
  GInputStream *input_stream;
  g_autofree char *builddir = NULL;
  g_autofree char *meson = NULL;
  JsonNode *root;

  g_assert (PLUGIN_IS_MESON_INTROSPECTION_STAGE (self));

  builddir = plugin_meson_base_stage_dup_builddir (PLUGIN_MESON_BASE_STAGE (stage));
  meson = plugin_meson_base_stage_dup_meson (PLUGIN_MESON_BASE_STAGE (stage));
  pipeline = foundry_build_stage_dup_pipeline (stage);

  g_assert (builddir != NULL);
  g_assert (meson != NULL);
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));

  launcher = foundry_process_launcher_new ();
  if (!dex_await (foundry_build_pipeline_prepare (pipeline, launcher, FOUNDRY_BUILD_PIPELINE_PHASE_CONFIGURE), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  foundry_process_launcher_set_cwd (launcher, builddir);
  foundry_process_launcher_append_argv (launcher, meson);
  foundry_process_launcher_append_argv (launcher, "introspect");
  foundry_process_launcher_append_argv (launcher, "--all");
  foundry_process_launcher_append_argv (launcher, "--force-object-output");

  if (!(io_stream = foundry_process_launcher_create_stdio_stream (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  dex_future_disown (dex_subprocess_wait_check (subprocess));

  input_stream = g_io_stream_get_input_stream (io_stream);
  parser = json_parser_new ();

  if (!dex_await (foundry_json_parser_load_from_stream (parser, input_stream), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  root = json_parser_get_root (parser);

  g_assert (root != NULL);

  g_clear_pointer (&self->introspection, json_node_unref);
  self->introspection = json_node_ref (root);

  return dex_future_new_true ();
}

static DexFuture *
plugin_meson_introspection_stage_build (FoundryBuildStage    *stage,
                                        FoundryBuildProgress *progress)
{
  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_meson_introspection_stage_build_fiber),
                                  2,
                                  FOUNDRY_TYPE_BUILD_STAGE, stage,
                                  FOUNDRY_TYPE_BUILD_PROGRESS, progress);
}

static DexFuture *
plugin_meson_introspection_stage_list_build_targets_fiber (gpointer data)
{
  g_autoptr(GListStore) store = NULL;
  JsonNode *root = data;
  JsonObject *root_obj;
  JsonArray *targets_ar;
  JsonNode *targets;

  g_assert (root != NULL);

  store = g_list_store_new (FOUNDRY_TYPE_BUILD_TARGET);

  if (JSON_NODE_HOLDS_OBJECT (root) &&
      (root_obj = json_node_get_object (root)) &&
      json_object_has_member (root_obj, "targets") &&
      (targets = json_object_get_member (root_obj, "targets")) &&
      JSON_NODE_HOLDS_ARRAY (targets) &&
      (targets_ar = json_node_get_array (targets)))
    {
      guint length = json_array_get_length (targets_ar);

      for (guint i = 0; i < length; i++)
        {
          JsonNode *target = json_array_get_element (targets_ar, i);
          JsonObject *target_obj;

          if (JSON_NODE_HOLDS_OBJECT (target) &&
              (target_obj = json_node_get_object (target)))
            {
              g_autoptr(FoundryBuildTarget) build_target = plugin_meson_build_target_new (target);

              if (build_target != NULL)
                g_list_store_append (store, build_target);
            }
        }
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static DexFuture *
plugin_meson_introspection_stage_list_build_targets (FoundryBuildStage *stage)
{
  PluginMesonIntrospectionStage *self = PLUGIN_MESON_INTROSPECTION_STAGE (stage);

  if (self->introspection == NULL)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "No introspection data available");

  return dex_scheduler_spawn (NULL, 0,
                              plugin_meson_introspection_stage_list_build_targets_fiber,
                              json_node_ref (self->introspection),
                              (GDestroyNotify) json_node_unref);

  return NULL;
}

static void
plugin_meson_introspection_stage_finalize (GObject *object)
{
  PluginMesonIntrospectionStage *self = (PluginMesonIntrospectionStage *)object;

  g_clear_pointer (&self->introspection, json_node_unref);

  G_OBJECT_CLASS (plugin_meson_introspection_stage_parent_class)->finalize (object);
}

static void
plugin_meson_introspection_stage_class_init (PluginMesonIntrospectionStageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryBuildStageClass *build_stage_class = FOUNDRY_BUILD_STAGE_CLASS (klass);

  object_class->finalize = plugin_meson_introspection_stage_finalize;

  build_stage_class->get_phase = plugin_meson_introspection_stage_get_phase;
  build_stage_class->build = plugin_meson_introspection_stage_build;
  build_stage_class->list_build_targets = plugin_meson_introspection_stage_list_build_targets;
}

static void
plugin_meson_introspection_stage_init (PluginMesonIntrospectionStage *self)
{
}

static DexFuture *
plugin_meson_introspection_stage_list_tests_fiber (FoundryContext *context,
                                                   JsonNode       *root,
                                                   const char     *builddir)
{
  g_autoptr(GListStore) store = NULL;
  JsonObject *root_obj;
  JsonArray *member_ar;
  JsonNode *member;

  g_assert (FOUNDRY_IS_CONTEXT (context));
  g_assert (root != NULL);

  store = g_list_store_new (FOUNDRY_TYPE_TEST);

  if (JSON_NODE_HOLDS_OBJECT (root) &&
      (root_obj = json_node_get_object (root))  &&
      json_object_has_member (root_obj, "tests") &&
      (member = json_object_get_member (root_obj, "tests")) &&
      JSON_NODE_HOLDS_ARRAY (member) &&
      (member_ar = json_node_get_array (member)))
    {
      guint length = json_array_get_length (member_ar);

      for (guint i = 0; i < length; i++)
        {
          JsonNode *element = json_array_get_element (member_ar, i);

          if (JSON_NODE_HOLDS_OBJECT (element))
            {
              g_autoptr(PluginMesonTest) test = plugin_meson_test_new (context, element);

              g_list_store_append (store, test);
            }
        }
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

DexFuture *
plugin_meson_introspection_stage_list_tests (PluginMesonIntrospectionStage *self)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autofree char *builddir = NULL;

  dex_return_error_if_fail (PLUGIN_IS_MESON_INTROSPECTION_STAGE (self));

  if (self->introspection == NULL)
    return foundry_future_new_not_supported ();

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  builddir = plugin_meson_base_stage_dup_builddir (PLUGIN_MESON_BASE_STAGE (self));

  return foundry_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                                  G_CALLBACK (plugin_meson_introspection_stage_list_tests_fiber),
                                  2,
                                  FOUNDRY_TYPE_CONTEXT, context,
                                  JSON_TYPE_NODE, self->introspection,
                                  G_TYPE_STRING, builddir);
}
