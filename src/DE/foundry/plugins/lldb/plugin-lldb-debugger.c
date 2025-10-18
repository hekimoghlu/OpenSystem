/* plugin-lldb-debugger.c
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

#include <locale.h>

#include "plugin-lldb-debugger.h"

struct _PluginLldbDebugger
{
  FoundryDapDebugger parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginLldbDebugger, plugin_lldb_debugger, FOUNDRY_TYPE_DAP_DEBUGGER)

static gboolean
environ_parse (const char  *pair,
               char       **key,
               char       **value)
{
  const char *eq;

  g_assert (pair != NULL);

  if (key != NULL)
    *key = NULL;

  if (value != NULL)
    *value = NULL;

  if ((eq = strchr (pair, '=')))
    {
      if (key != NULL)
        *key = g_strndup (pair, eq - pair);

      if (value != NULL)
        *value = g_strdup (eq + 1);

      return TRUE;
    }

  return FALSE;
}

static JsonNode *
env_to_object (const char * const *env)
{
  {
    g_autoptr(JsonObject)object = json_object_new ();
    JsonNode *node = json_node_new (JSON_NODE_OBJECT);

    if (env != NULL)
      {
        for (guint i = 0; env[i]; i++)
          {
            g_autofree char *key = NULL;
            g_autofree char *value = NULL;

            if (environ_parse (env[i], &key, &value))
              json_object_set_string_member (object, key, value);
          }
      }

    json_node_set_object (node, object);

    return node;
  }
}

static DexFuture *
plugin_lldb_debugger_call_checked (PluginLldbDebugger *self,
                                   JsonNode           *node)
{
  return dex_future_then (foundry_dap_debugger_call (FOUNDRY_DAP_DEBUGGER (self), node),
                          foundry_dap_protocol_unwrap_error,
                          NULL, NULL);
}

static DexFuture *
plugin_lldb_debugger_connect_to_target_fiber (PluginLldbDebugger    *self,
                                              FoundryDebuggerTarget *target)
{
  g_autoptr(DexFuture) launch = NULL;
  g_autoptr(JsonNode) reply = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (PLUGIN_IS_LLDB_DEBUGGER (self));
  g_assert (FOUNDRY_IS_DEBUGGER_TARGET (target));

  if (FOUNDRY_IS_DEBUGGER_TARGET_COMMAND (target))
    {
      g_autoptr(FoundryCommand) command = NULL;

      if ((command = foundry_debugger_target_command_dup_command (FOUNDRY_DEBUGGER_TARGET_COMMAND (target))))
        {
          g_autofree char *cwd = foundry_command_dup_cwd (command);
          g_auto(GStrv) argv = foundry_command_dup_argv (command);
          g_auto(GStrv) env = foundry_command_dup_environ (command);
          g_autoptr(JsonNode) env_object = env_to_object ((const char * const *)env);

          launch = foundry_dap_debugger_call (FOUNDRY_DAP_DEBUGGER (self),
                                              FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                       "command", "launch",
                                                                       "arguments", "{",
                                                                         "noDebug", FOUNDRY_JSON_NODE_PUT_BOOLEAN (FALSE),
                                                                         "args", FOUNDRY_JSON_NODE_PUT_STRV ((const char * const *)argv),
                                                                         "program", FOUNDRY_JSON_NODE_PUT_STRING (argv[0]),
                                                                         "env", FOUNDRY_JSON_NODE_PUT_NODE (env_object),
                                                                         "cwd", FOUNDRY_JSON_NODE_PUT_STRING (cwd),
                                                                         "stopOnEntry", FOUNDRY_JSON_NODE_PUT_BOOLEAN (FALSE),
                                                                         "disableASLR", FOUNDRY_JSON_NODE_PUT_BOOLEAN (FALSE),
                                                                       "}"));
        }
    }
  else if (FOUNDRY_IS_DEBUGGER_TARGET_PROCESS (target))
    {
      GPid pid = foundry_debugger_target_process_get_pid (FOUNDRY_DEBUGGER_TARGET_PROCESS (target));

      launch = foundry_dap_debugger_call (FOUNDRY_DAP_DEBUGGER (self),
                                          FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                   "command", "attach",
                                                                   "arguments", "{",
                                                                     "pid", FOUNDRY_JSON_NODE_PUT_INT (pid),
                                                                     "program", FOUNDRY_JSON_NODE_PUT_STRING (NULL),
                                                                     "stopOnAttach", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
                                                                   "}"));
    }
  else if (FOUNDRY_IS_DEBUGGER_TARGET_REMOTE (target))
    {
      g_autofree char *address = foundry_debugger_target_remote_dup_address (FOUNDRY_DEBUGGER_TARGET_REMOTE (target));

      launch = foundry_dap_debugger_call (FOUNDRY_DAP_DEBUGGER (self),
                                          FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                   "command", "attach",
                                                                   "arguments", "{",
                                                                     "connect", FOUNDRY_JSON_NODE_PUT_STRING (address),
                                                                     "program", FOUNDRY_JSON_NODE_PUT_STRING (NULL),
                                                                     "stopOnAttach", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
                                                                   "}"));

      return dex_future_new_true ();
    }

  if (launch == NULL)
    return foundry_future_new_not_supported ();

  /* We have to send our configurationDone after our launch because
   * our launch/attach will not complete until configurationDone is
   * called. But we also can't call configurationDone until after the
   * launch/attach has been called (but not yet returned).
   */
  dex_future_disown (foundry_dap_debugger_send (FOUNDRY_DAP_DEBUGGER (self),
                                                FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                         "command", "configurationDone")));

  if (!(reply = dex_await_boxed (dex_ref (launch), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (foundry_dap_protocol_has_error (reply))
    return dex_future_new_for_error (foundry_dap_protocol_extract_error (reply));

  return dex_future_new_true ();
}

static DexFuture *
plugin_lldb_debugger_connect_to_target (FoundryDebugger       *debugger,
                                        FoundryDebuggerTarget *target)
{
  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_lldb_debugger_connect_to_target_fiber),
                                  2,
                                  PLUGIN_TYPE_LLDB_DEBUGGER, debugger,
                                  FOUNDRY_TYPE_DEBUGGER_TARGET, target);
}

static DexFuture *
plugin_lldb_debugger_initialize_fiber (gpointer data)
{
  PluginLldbDebugger *self = data;
  g_autoptr(FoundryDebuggerTrapParams) params = NULL;
  g_autoptr(JsonNode) reply = NULL;
  g_autoptr(JsonNode) message = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (PLUGIN_IS_LLDB_DEBUGGER (self));

  message = FOUNDRY_JSON_OBJECT_NEW (
    "type", "request",
    "command", "initialize",
    "arguments", "{",
      "clientID", "libfoundry-" PACKAGE_VERSION,
      "clientName", FOUNDRY_JSON_NODE_PUT_STRING (g_get_application_name ()),
      "adapterID", "libfoundry-" PACKAGE_VERSION,
      "locale", FOUNDRY_JSON_NODE_PUT_STRING (setlocale (LC_ALL, NULL)),
      "pathFormat", "uri",
      "columnsStartAt1", FOUNDRY_JSON_NODE_PUT_BOOLEAN (FALSE),
      "linesStartAt1", FOUNDRY_JSON_NODE_PUT_BOOLEAN (FALSE),
      "supportsANSIStyling", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
      "supportsArgsCanBeInterpretedByShell", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
      "supportsMemoryEvent", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
      "supportsMemoryReferences", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
      "supportsProgressReporting", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
      "supportsRunInTerminalRequest", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
      "supportsStartDebuggingRequest", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
      "supportsVariablePaging", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
      "supportsVariableType", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
    "}"
  );

  if (!(reply = dex_await_boxed (plugin_lldb_debugger_call_checked (self, g_steal_pointer (&message)), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  params = foundry_debugger_trap_params_new ();
  foundry_debugger_trap_params_set_function (params, "main");
  foundry_debugger_trap_params_set_kind (params, FOUNDRY_DEBUGGER_TRAP_KIND_BREAKPOINT);

  if (!dex_await (foundry_debugger_trap (FOUNDRY_DEBUGGER (self), params), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static DexFuture *
plugin_lldb_debugger_initialize (FoundryDebugger *debugger)
{
  PluginLldbDebugger *self = (PluginLldbDebugger *)debugger;

  g_assert (PLUGIN_IS_LLDB_DEBUGGER (self));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_lldb_debugger_initialize_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static char *
plugin_lldb_debugger_dup_name (FoundryDebugger *debugger)
{
  return g_strdup ("lldb");
}

static void
plugin_lldb_debugger_class_init (PluginLldbDebuggerClass *klass)
{
  FoundryDebuggerClass *debugger_class = FOUNDRY_DEBUGGER_CLASS (klass);

  debugger_class->connect_to_target = plugin_lldb_debugger_connect_to_target;
  debugger_class->dup_name = plugin_lldb_debugger_dup_name;
  debugger_class->initialize = plugin_lldb_debugger_initialize;
}

static void
plugin_lldb_debugger_init (PluginLldbDebugger *self)
{
}

FoundryDebugger *
plugin_lldb_debugger_new (FoundryContext *context,
                          GSubprocess    *subprocess,
                          GIOStream      *stream)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (G_IS_SUBPROCESS (subprocess), NULL);
  g_return_val_if_fail (G_IS_IO_STREAM (stream), NULL);

  return g_object_new (PLUGIN_TYPE_LLDB_DEBUGGER,
                       "quirks", FOUNDRY_DAP_DEBUGGER_QUIRK_QUERY_THREADS,
                       "context", context,
                       "subprocess", subprocess,
                       "stream", stream,
                       NULL);
}
