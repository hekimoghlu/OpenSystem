/* plugin-meson-test-provider.c
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

#include "plugin-meson-introspection-stage.h"
#include "plugin-meson-test-provider.h"

struct _PluginMesonTestProvider
{
  FoundryTestProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginMesonTestProvider, plugin_meson_test_provider, FOUNDRY_TYPE_TEST_PROVIDER)

static DexFuture *
plugin_meson_test_provider_load (FoundryTestProvider *provider)
{
  PluginMesonTestProvider *self = (PluginMesonTestProvider *)provider;

  g_assert (PLUGIN_IS_MESON_TEST_PROVIDER (self));

  return dex_future_new_true ();
}

static DexFuture *
plugin_meson_test_provider_unload (FoundryTestProvider *provider)
{
  PluginMesonTestProvider *self = (PluginMesonTestProvider *)provider;

  g_assert (PLUGIN_IS_MESON_TEST_PROVIDER (self));

  return dex_future_new_true ();
}

static DexFuture *
plugin_meson_test_provider_list_tests (FoundryTestProvider *provider)
{
  PluginMesonTestProvider *self = (PluginMesonTestProvider *)provider;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildProgress) progress = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GError) error = NULL;
  guint n_items;

  g_assert (PLUGIN_IS_MESON_TEST_PROVIDER (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  build_manager = foundry_context_dup_build_manager (context);
  cancellable = dex_cancellable_new ();

  /* Get the pipeline to ensure our introspection stage ran */
  if (!(pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  /* Advance progress to (CONFIGURE|AFTER) if necessary */
  progress = foundry_build_pipeline_build (pipeline,
                                           FOUNDRY_BUILD_PIPELINE_PHASE_CONFIGURE,
                                           -1, cancellable);
  if (!dex_await (foundry_build_progress_await (progress), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  /* Locate introspection stage and ask it for a list of tests */
  n_items = g_list_model_get_n_items (G_LIST_MODEL (pipeline));
  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryBuildStage) stage = g_list_model_get_item (G_LIST_MODEL (pipeline), i);

      if (PLUGIN_IS_MESON_INTROSPECTION_STAGE (stage))
        return plugin_meson_introspection_stage_list_tests (PLUGIN_MESON_INTROSPECTION_STAGE (stage));
    }

  g_warning ("Failed to locate `%s` in pipeline",
             g_type_name (PLUGIN_TYPE_MESON_INTROSPECTION_STAGE));

  return dex_future_new_take_object (g_list_store_new (FOUNDRY_TYPE_TEST));
}

static void
plugin_meson_test_provider_class_init (PluginMesonTestProviderClass *klass)
{
  FoundryTestProviderClass *test_provider_class = FOUNDRY_TEST_PROVIDER_CLASS (klass);

  test_provider_class->load = plugin_meson_test_provider_load;
  test_provider_class->unload = plugin_meson_test_provider_unload;
  test_provider_class->list_tests = plugin_meson_test_provider_list_tests;
}

static void
plugin_meson_test_provider_init (PluginMesonTestProvider *self)
{
}
