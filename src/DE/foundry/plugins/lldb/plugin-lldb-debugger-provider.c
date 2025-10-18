/* plugin-lldb-debugger-provider.c
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

#include "plugin-lldb-debugger.h"
#include "plugin-lldb-debugger-provider.h"

struct _PluginLldbDebuggerProvider
{
  FoundryDebuggerProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginLldbDebuggerProvider, plugin_lldb_debugger_provider, FOUNDRY_TYPE_DEBUGGER_PROVIDER)

static DexFuture *
plugin_lldb_debugger_provider_load_debugger_fiber (FoundryDebuggerProvider *provider,
                                                  FoundryBuildPipeline    *pipeline)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GIOStream) io_stream = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_DEBUGGER_PROVIDER (provider));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (provider));
  launcher = foundry_process_launcher_new ();

  if (pipeline != NULL)
    {
      if (!dex_await (foundry_build_pipeline_prepare (pipeline, launcher, FOUNDRY_BUILD_PIPELINE_PHASE_BUILD), &error))
        return dex_future_new_for_error (g_steal_pointer (&error));
    }

  foundry_process_launcher_append_argv (launcher, "lldb-dap");

  if (!(io_stream = foundry_process_launcher_create_stdio_stream (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_take_object (plugin_lldb_debugger_new (context, subprocess, io_stream));
}

static DexFuture *
plugin_lldb_debugger_provider_load_debugger (FoundryDebuggerProvider *provider,
                                            FoundryBuildPipeline    *pipeline)
{
  g_assert (FOUNDRY_IS_DEBUGGER_PROVIDER (provider));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_lldb_debugger_provider_load_debugger_fiber),
                                  2,
                                  FOUNDRY_TYPE_DEBUGGER_PROVIDER, provider,
                                  FOUNDRY_TYPE_BUILD_PIPELINE, pipeline);
}

static DexFuture *
plugin_lldb_debugger_provider_supports_fiber (FoundryDebuggerProvider *provider,
                                             FoundryBuildPipeline    *pipeline,
                                             FoundryCommand          *command)
{
  g_assert (PLUGIN_IS_LLDB_DEBUGGER_PROVIDER (provider));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_COMMAND (command));

  g_debug ("Looking for lldb in pipeline %p", pipeline);

  if (pipeline != NULL)
    {
      g_autoptr(GError) error = NULL;

      if (!dex_await (foundry_build_pipeline_contains_program (pipeline, "lldb-dap"), &error))
        {
          g_debug ("`lldb-dap` was not found: %s", error->message);
          return dex_future_new_for_error (g_steal_pointer (&error));
        }
    }

  return dex_future_new_for_int (10);
}

static DexFuture *
plugin_lldb_debugger_provider_supports (FoundryDebuggerProvider *provider,
                                       FoundryBuildPipeline    *pipeline,
                                       FoundryCommand          *command)
{
  g_assert (PLUGIN_IS_LLDB_DEBUGGER_PROVIDER (provider));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_COMMAND (command));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_lldb_debugger_provider_supports_fiber),
                                  3,
                                  FOUNDRY_TYPE_DEBUGGER_PROVIDER, provider,
                                  FOUNDRY_TYPE_BUILD_PIPELINE, pipeline,
                                  FOUNDRY_TYPE_COMMAND, command);
}

static void
plugin_lldb_debugger_provider_class_init (PluginLldbDebuggerProviderClass *klass)
{
  FoundryDebuggerProviderClass *debugger_provider_class = FOUNDRY_DEBUGGER_PROVIDER_CLASS (klass);

  debugger_provider_class->supports = plugin_lldb_debugger_provider_supports;
  debugger_provider_class->load_debugger = plugin_lldb_debugger_provider_load_debugger;
}

static void
plugin_lldb_debugger_provider_init (PluginLldbDebuggerProvider *self)
{
}
