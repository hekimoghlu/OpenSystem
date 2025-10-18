/* plugin-gdb-debugger.c
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
#include <stdio.h>

#include "line-reader-private.h"

#include "plugin-gdb-debugger.h"
#include "plugin-gdb-mapped-region.h"

struct _PluginGdbDebugger
{
  FoundryDapDebugger  parent_instance;
  GListStore         *mappings;
};

G_DEFINE_FINAL_TYPE (PluginGdbDebugger, plugin_gdb_debugger, FOUNDRY_TYPE_DAP_DEBUGGER)

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
  if (env == NULL || env[0] == NULL)
    return json_node_new (JSON_NODE_NULL);

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
plugin_gdb_debugger_call_checked (PluginGdbDebugger *self,
                                  JsonNode          *node)
{
  return dex_future_then (foundry_dap_debugger_call (FOUNDRY_DAP_DEBUGGER (self), node),
                          foundry_dap_protocol_unwrap_error,
                          NULL, NULL);
}

static DexFuture *
plugin_gdb_debugger_connect_to_target_fiber (PluginGdbDebugger     *self,
                                             FoundryDebuggerTarget *target)
{
  g_autoptr(DexFuture) launch = NULL;
  g_autoptr(JsonNode) reply = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (PLUGIN_IS_GDB_DEBUGGER (self));
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
                                                                         "args", FOUNDRY_JSON_NODE_PUT_STRV ((const char * const *)&argv[1]),
                                                                         "program", FOUNDRY_JSON_NODE_PUT_STRING (argv[0]),
                                                                         "env", FOUNDRY_JSON_NODE_PUT_NODE (env_object),
                                                                         "cwd", FOUNDRY_JSON_NODE_PUT_STRING (cwd),
                                                                         "stopAtBeginningOfMainSubprogram", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
                                                                         "stopOnEntry", FOUNDRY_JSON_NODE_PUT_BOOLEAN (FALSE),
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
                                                                   "}"));
    }
  else if (FOUNDRY_IS_DEBUGGER_TARGET_REMOTE (target))
    {
      g_autofree char *address = foundry_debugger_target_remote_dup_address (FOUNDRY_DEBUGGER_TARGET_REMOTE (target));

      launch = foundry_dap_debugger_call (FOUNDRY_DAP_DEBUGGER (self),
                                          FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                   "command", "attach",
                                                                   "arguments", "{",
                                                                     "target", FOUNDRY_JSON_NODE_PUT_STRING (address),
                                                                     "program", FOUNDRY_JSON_NODE_PUT_STRING (NULL),
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
plugin_gdb_debugger_connect_to_target (FoundryDebugger       *debugger,
                                       FoundryDebuggerTarget *target)
{
  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_gdb_debugger_connect_to_target_fiber),
                                  2,
                                  PLUGIN_TYPE_GDB_DEBUGGER, debugger,
                                  FOUNDRY_TYPE_DEBUGGER_TARGET, target);
}

static DexFuture *
plugin_gdb_debugger_initialize_fiber (gpointer data)
{
  PluginGdbDebugger *self = data;
  g_autoptr(FoundryDebuggerTrapParams) params = NULL;
  g_autoptr(JsonNode) reply = NULL;
  g_autoptr(JsonNode) message = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (PLUGIN_IS_GDB_DEBUGGER (self));

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

  if (!(reply = dex_await_boxed (plugin_gdb_debugger_call_checked (self, g_steal_pointer (&message)), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  params = foundry_debugger_trap_params_new ();
  foundry_debugger_trap_params_set_function (params, "main");
  foundry_debugger_trap_params_set_kind (params, FOUNDRY_DEBUGGER_TRAP_KIND_BREAKPOINT);

  if (!dex_await (foundry_debugger_trap (FOUNDRY_DEBUGGER (self), params), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static DexFuture *
plugin_gdb_debugger_initialize (FoundryDebugger *debugger)
{
  PluginGdbDebugger *self = (PluginGdbDebugger *)debugger;

  g_assert (PLUGIN_IS_GDB_DEBUGGER (self));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_gdb_debugger_initialize_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static char *
plugin_gdb_debugger_dup_name (FoundryDebugger *debugger)
{
  return g_strdup ("gdb");
}

static DexFuture *
plugin_gdb_debugger_send_signal (FoundryDebugger *debugger,
                                 int              signum)
{
  g_autofree char *command = NULL;

  dex_return_error_if_fail (PLUGIN_IS_GDB_DEBUGGER (debugger));

  command = g_strdup_printf ("signal %d", signum);

  return foundry_debugger_interpret (debugger, command);
}

static DexFuture *
plugin_gdb_debugger_parse_mappings (DexFuture *completed,
                                    gpointer   user_data)
{
  PluginGdbDebugger *self = user_data;
  g_autoptr(JsonNode) node = NULL;
  const char *input;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (PLUGIN_IS_GDB_DEBUGGER (self));

  if ((node = dex_await_boxed (dex_ref (completed), NULL)) &&
      FOUNDRY_JSON_OBJECT_PARSE (node,
                                 "body", "{",
                                   "result", FOUNDRY_JSON_NODE_GET_STRING (&input),
                                 "}"))
    {
      g_autofree char *copy = g_strdup (input);
      g_autoptr(GPtrArray) regions = NULL;
      LineReader reader;
      char *line;
      gsize len;

      regions = g_ptr_array_new_with_free_func (g_object_unref);

      line_reader_init (&reader, copy, -1);
      while ((line = line_reader_next (&reader, &len)))
        {
          g_autofree char *path = NULL;
          guint64 begin = 0;
          guint64 end = 0;
          guint64 size = 0;
          guint64 offset = 0;
          guint mode = 0;
          char rd = 0;
          char wr = 0;
          char ex = 0;
          char p = 0;

          line[len] = 0;

          if (len < 32)
            continue;

          /* Just make sure path is long enough to hold anything
           * on the line. This is the last field so it is guaranteed
           * to be less than the length of the line.
           */
          path = g_new0 (char, len + 1);

          if (sscanf (line,
                      "0x%"G_GINT64_MODIFIER"x "
                      "0x%"G_GINT64_MODIFIER"x "
                      "0x%"G_GINT64_MODIFIER"x "
                      "0x%"G_GINT64_MODIFIER"x "
                      "%c%c%c%c "
                      "%s",
                      &begin, &end, &size, &offset,
                      &rd, &wr, &ex, &p, path) != 9)
            continue;

          if (rd == 'r') mode |= 4;
          if (wr == 'w') mode |= 2;
          if (ex == 'x') mode |= 1;

          path[len] = 0;

          g_ptr_array_add (regions, plugin_gdb_mapped_region_new (begin, end, offset, mode, path));
        }

      if (regions->len || g_list_model_get_n_items (G_LIST_MODEL (self->mappings)))
        g_list_store_splice (self->mappings,
                             0,
                             g_list_model_get_n_items (G_LIST_MODEL (self->mappings)),
                             regions->pdata,
                             regions->len);
    }

  return dex_future_new_true ();
}

static void
plugin_gdb_debugger_event (FoundryDebugger      *debugger,
                           FoundryDebuggerEvent *event)
{
  PluginGdbDebugger *self = PLUGIN_GDB_DEBUGGER (debugger);

  g_assert (PLUGIN_IS_GDB_DEBUGGER (self));
  g_assert (FOUNDRY_IS_DEBUGGER_EVENT (event));

  if (FOUNDRY_IS_DEBUGGER_STOP_EVENT (event))
    {
      DexFuture *future;

      /* DAP modules do not include address ranges in notifications and
       * when querying them they do not include the full area where all
       * sections are mapped.
       *
       * Instead, with gdb we can use a specific query to get that
       * information for the user but it needs to be updated when we
       * stop since it could have changed.
       */

      future = foundry_debugger_interpret (debugger, "info proc mappings");
      future = dex_future_then (future,
                                plugin_gdb_debugger_parse_mappings,
                                g_object_ref (self),
                                g_object_unref);

      dex_future_disown (future);
    }
}

static GListModel *
plugin_gdb_debugger_list_address_space (FoundryDebugger *debugger)
{
  return g_object_ref (G_LIST_MODEL (PLUGIN_GDB_DEBUGGER (debugger)->mappings));
}

static void
plugin_gdb_debugger_finalize (GObject *object)
{
  PluginGdbDebugger *self = (PluginGdbDebugger *)object;

  g_clear_object (&self->mappings);

  G_OBJECT_CLASS (plugin_gdb_debugger_parent_class)->finalize (object);
}

static void
plugin_gdb_debugger_class_init (PluginGdbDebuggerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDebuggerClass *debugger_class = FOUNDRY_DEBUGGER_CLASS (klass);

  object_class->finalize = plugin_gdb_debugger_finalize;

  debugger_class->connect_to_target = plugin_gdb_debugger_connect_to_target;
  debugger_class->dup_name = plugin_gdb_debugger_dup_name;
  debugger_class->initialize = plugin_gdb_debugger_initialize;
  debugger_class->send_signal = plugin_gdb_debugger_send_signal;
  debugger_class->event = plugin_gdb_debugger_event;
  debugger_class->list_address_space = plugin_gdb_debugger_list_address_space;
}

static void
plugin_gdb_debugger_init (PluginGdbDebugger *self)
{
  self->mappings = g_list_store_new (FOUNDRY_TYPE_DEBUGGER_MAPPED_REGION);
}

FoundryDebugger *
plugin_gdb_debugger_new (FoundryContext *context,
                         GSubprocess    *subprocess,
                         GIOStream      *stream)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (G_IS_SUBPROCESS (subprocess), NULL);
  g_return_val_if_fail (G_IS_IO_STREAM (stream), NULL);

  return g_object_new (PLUGIN_TYPE_GDB_DEBUGGER,
                       "context", context,
                       "subprocess", subprocess,
                       "stream", stream,
                       NULL);
}
