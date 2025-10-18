/* plugin-llm-subprocess-tool.c
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

#include "plugin-llm-subprocess-tool.h"

struct _PluginLlmSubprocessTool
{
  FoundryLlmTool parent_instance;
  char *name;
  char *description;
  char **argv;
};

G_DEFINE_FINAL_TYPE (PluginLlmSubprocessTool, plugin_llm_subprocess_tool, FOUNDRY_TYPE_LLM_TOOL)

static char *
plugin_llm_subprocess_tool_dup_name (FoundryLlmTool *tool)
{
  return g_strdup (PLUGIN_LLM_SUBPROCESS_TOOL (tool)->name);
}

static char *
plugin_llm_subprocess_tool_dup_description (FoundryLlmTool *tool)
{
  return g_strdup (PLUGIN_LLM_SUBPROCESS_TOOL (tool)->description);
}

static DexFuture *
plugin_llm_subprocess_tool_call_fiber (gpointer data)
{
  PluginLlmSubprocessTool *self = data;
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryDBusService) dbus = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) project_dir = NULL;
  g_autofree char *output = NULL;
  g_autofree char *address = NULL;
  g_autofree char *stdout_buf = NULL;

  g_assert (PLUGIN_IS_LLM_SUBPROCESS_TOOL (self));

  if (!(context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    return foundry_future_new_disposed ();

  launcher = foundry_process_launcher_new ();

  project_dir = foundry_context_dup_project_directory (context);
  foundry_process_launcher_set_cwd (launcher, g_file_peek_path (project_dir));
  foundry_process_launcher_append_args (launcher, (const char * const *)self->argv);

  dbus = foundry_context_dup_dbus_service (context);

  if (!(address = dex_await_string (foundry_dbus_service_query_address (dbus), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (address != NULL)
    foundry_process_launcher_setenv (launcher, "FOUNDRY_ADDRESS", address);

  if (!(subprocess = foundry_process_launcher_spawn_with_flags (launcher,
                                                                (G_SUBPROCESS_FLAGS_STDOUT_PIPE |
                                                                 G_SUBPROCESS_FLAGS_STDERR_MERGE),
                                                                &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(stdout_buf = dex_await_string (foundry_subprocess_communicate_utf8 (subprocess, NULL), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_take_object (foundry_simple_llm_message_new (g_strdup ("tool"),
                                                                     g_steal_pointer (&stdout_buf)));
}

static DexFuture *
plugin_llm_subprocess_tool_call (FoundryLlmTool *tool,
                                 const GValue   *params,
                                 guint           n_params)
{
  dex_return_error_if_fail (PLUGIN_IS_LLM_SUBPROCESS_TOOL (tool));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_llm_subprocess_tool_call_fiber,
                              g_object_ref (tool),
                              g_object_unref);
}

static void
plugin_llm_subprocess_tool_finalize (GObject *object)
{
  PluginLlmSubprocessTool *self = (PluginLlmSubprocessTool *)object;

  g_clear_pointer (&self->name, g_free);
  g_clear_pointer (&self->description, g_free);
  g_clear_pointer (&self->argv, g_strfreev);

  G_OBJECT_CLASS (plugin_llm_subprocess_tool_parent_class)->finalize (object);
}

static void
plugin_llm_subprocess_tool_class_init (PluginLlmSubprocessToolClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryLlmToolClass *llm_tool_class = FOUNDRY_LLM_TOOL_CLASS (klass);

  object_class->finalize = plugin_llm_subprocess_tool_finalize;

  llm_tool_class->dup_name = plugin_llm_subprocess_tool_dup_name;
  llm_tool_class->dup_description = plugin_llm_subprocess_tool_dup_description;
  llm_tool_class->call = plugin_llm_subprocess_tool_call;
}

static void
plugin_llm_subprocess_tool_init (PluginLlmSubprocessTool *self)
{
}

FoundryLlmTool *
plugin_llm_subprocess_tool_new (FoundryContext     *context,
                                const char         *name,
                                const char * const *argv,
                                const char         *description)
{
  PluginLlmSubprocessTool *self;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (name != NULL, NULL);
  g_return_val_if_fail (argv != NULL, NULL);

  self = g_object_new (PLUGIN_TYPE_LLM_SUBPROCESS_TOOL,
                       "context", context,
                       NULL);
  self->name = g_strdup (name);
  self->argv = g_strdupv ((char **)argv);
  self->description = g_strdup (description);

  return FOUNDRY_LLM_TOOL (self);
}
