/* plugin-llm-build-provider.c
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

#include "plugin-llm-build-provider.h"
#include "plugin-llm-subprocess-tool.h"

struct _PluginLlmBuildProvider
{
  FoundryLlmProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginLlmBuildProvider, plugin_llm_build_provider, FOUNDRY_TYPE_LLM_PROVIDER)

static const struct {
  const char *name;
  const char * const *argv;
  const char *description;
} tool_infos[] = {
  {
    "build",
    FOUNDRY_STRV_INIT ("foundry", "pipeline", "build"),
    "Incrementally build the project to provide any new diagnostics or build failures",
  },
  {
    "rebuild",
    FOUNDRY_STRV_INIT ("foundry", "pipeline", "rebuild"),
    "Rebuild the project from scratch which can help elevate additional diagnostics",
  },
};

static DexFuture *
plugin_llm_build_provider_list_tools (FoundryLlmProvider *provider)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListStore) store = NULL;

  g_assert (PLUGIN_IS_LLM_BUILD_PROVIDER (provider));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (provider));
  store = g_list_store_new (FOUNDRY_TYPE_LLM_TOOL);

  for (guint i = 0; i < G_N_ELEMENTS (tool_infos); i++)
    {
      g_autoptr(FoundryLlmTool) tool = NULL;

      tool = plugin_llm_subprocess_tool_new (context,
                                             tool_infos[i].name,
                                             tool_infos[i].argv,
                                             tool_infos[i].description);
      g_list_store_append (store, tool);
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static void
plugin_llm_build_provider_class_init (PluginLlmBuildProviderClass *klass)
{
  FoundryLlmProviderClass *provider_class = FOUNDRY_LLM_PROVIDER_CLASS (klass);

  provider_class->list_tools = plugin_llm_build_provider_list_tools;
}

static void
plugin_llm_build_provider_init (PluginLlmBuildProvider *self)
{
}
