/* foundry-dap-debugger-variable.c
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

#include "foundry-dap-debugger-variable-private.h"
#include "foundry-dap-protocol.h"
#include "foundry-json-node.h"
#include "foundry-util.h"

struct _FoundryDapDebuggerVariable
{
  FoundryDebuggerVariable parent_instance;
  GWeakRef debugger_wr;
  JsonNode *node;
};

G_DEFINE_FINAL_TYPE (FoundryDapDebuggerVariable, foundry_dap_debugger_variable, FOUNDRY_TYPE_DEBUGGER_VARIABLE)

static char *
foundry_dap_debugger_variable_dup_name (FoundryDebuggerVariable *variable)
{
  FoundryDapDebuggerVariable *self = FOUNDRY_DAP_DEBUGGER_VARIABLE (variable);
  const char *name = NULL;

  if (FOUNDRY_JSON_OBJECT_PARSE (self->node, "name", FOUNDRY_JSON_NODE_GET_STRING (&name)))
    return g_strdup (name);

  return NULL;
}

static char *
foundry_dap_debugger_variable_dup_value (FoundryDebuggerVariable *variable)
{
  FoundryDapDebuggerVariable *self = FOUNDRY_DAP_DEBUGGER_VARIABLE (variable);
  const char *value = NULL;

  if (FOUNDRY_JSON_OBJECT_PARSE (self->node, "value", FOUNDRY_JSON_NODE_GET_STRING (&value)))
    return g_strdup (value);

  return NULL;
}

static char *
foundry_dap_debugger_variable_dup_type_name (FoundryDebuggerVariable *variable)
{
  FoundryDapDebuggerVariable *self = FOUNDRY_DAP_DEBUGGER_VARIABLE (variable);
  const char *type_name = NULL;

  if (FOUNDRY_JSON_OBJECT_PARSE (self->node, "type", FOUNDRY_JSON_NODE_GET_STRING (&type_name)))
    return g_strdup (type_name);

  return NULL;
}

static gboolean
foundry_dap_debugger_variable_is_structured (FoundryDebuggerVariable *variable,
                                             guint                   *n_children)
{
  FoundryDapDebuggerVariable *self = FOUNDRY_DAP_DEBUGGER_VARIABLE (variable);
  gboolean ret;
  gint64 reference_id = 0;
  gint64 count = 0;

  ret = (FOUNDRY_JSON_OBJECT_PARSE (self->node,
                                    "variablesReference", FOUNDRY_JSON_NODE_GET_INT (&reference_id)) &&
         reference_id > 0);

  if (FOUNDRY_JSON_OBJECT_PARSE (self->node,
                                 "namedVariables", FOUNDRY_JSON_NODE_GET_INT (&count)))
    *n_children = MIN (count, G_MAXUINT32);
  else if (FOUNDRY_JSON_OBJECT_PARSE (self->node,
                                      "indexedVariables", FOUNDRY_JSON_NODE_GET_INT (&count)))
    *n_children = MIN (count, G_MAXUINT32);

  return ret;
}

static DexFuture *
foundry_dap_debugger_variable_inflate_cb (DexFuture *completed,
                                          gpointer   user_data)
{
  FoundryDapDebuggerVariable * self = user_data;
  g_autoptr(FoundryDapDebugger) debugger = NULL;
  g_autoptr(JsonNode) variables_reply = NULL;
  g_autoptr(GListStore) store = NULL;
  JsonArray *variables_ar = NULL;
  JsonNode *variables = NULL;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (FOUNDRY_IS_DAP_DEBUGGER_VARIABLE (self));

  store = g_list_store_new (FOUNDRY_TYPE_DEBUGGER_VARIABLE);

  if ((debugger = g_weak_ref_get (&self->debugger_wr)) &&
      (variables_reply = dex_await_boxed (dex_ref (completed), NULL)) &&
      FOUNDRY_JSON_OBJECT_PARSE (variables_reply,
                                 "body", "{",
                                   "variables", FOUNDRY_JSON_NODE_GET_NODE (&variables),
                                 "}") &&
      JSON_NODE_HOLDS_ARRAY (variables) &&
      (variables_ar = json_node_get_array (variables)))
    {
      guint n_variables = json_array_get_length (variables_ar);

      for (guint v = 0; v < n_variables; v++)
        {
          JsonNode *variable_node = json_array_get_element (variables_ar, v);
          g_autoptr(FoundryDebuggerVariable) variable = NULL;

          if ((variable = foundry_dap_debugger_variable_new (debugger, variable_node)))
            g_list_store_append (store, variable);
        }
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static DexFuture *
foundry_dap_debugger_variable_list_children (FoundryDebuggerVariable *variable)
{
  FoundryDapDebuggerVariable *self = FOUNDRY_DAP_DEBUGGER_VARIABLE (variable);
  g_autoptr(FoundryDapDebugger) debugger = g_weak_ref_get (&self->debugger_wr);
  DexFuture *future;
  gint64 reference_id = 0;

  if (debugger == NULL)
    return foundry_future_new_disposed ();

  if (!FOUNDRY_JSON_OBJECT_PARSE (self->node,
                                  "variablesReference", FOUNDRY_JSON_NODE_GET_INT (&reference_id)))
    return foundry_future_new_not_supported ();

  future = foundry_dap_debugger_call (debugger,
                                      FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                               "command", "variables",
                                                               "arguments", "{",
                                                                 "variablesReference", FOUNDRY_JSON_NODE_PUT_INT (reference_id),
                                                               "}"));
  future = dex_future_then (future,
                            foundry_dap_protocol_unwrap_error,
                            NULL, NULL);
  future = dex_future_then (future,
                            foundry_dap_debugger_variable_inflate_cb,
                            g_object_ref (self),
                            g_object_unref);
  return future;
}

static DexFuture *
foundry_dap_debugger_variable_read_memory_cb (DexFuture *completed,
                                              gpointer   user_data)
{
  g_autoptr(JsonNode) node = NULL;
  const char *address = NULL;
  const char *base64 = NULL;

  node = dex_await_boxed (dex_ref (completed), NULL);

  if (FOUNDRY_JSON_OBJECT_PARSE (node,
                                 "type", "response",
                                 "command", "readMemory",
                                 "body", "{",
                                   "address", FOUNDRY_JSON_NODE_GET_STRING (&address),
                                   "data", FOUNDRY_JSON_NODE_GET_STRING (&base64),
                                 "}"))
    {
      gsize decoded_len = 0;
      guint8 *decoded = g_base64_decode (base64, &decoded_len);

      if (decoded != NULL)
        return dex_future_new_take_boxed (G_TYPE_BYTES, g_bytes_new (decoded, decoded_len));
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_INVALID_DATA,
                                "Invalid reply from peer");
}

static DexFuture *
foundry_dap_debugger_variable_read_memory (FoundryDebuggerVariable *variable,
                                           guint64                  offset,
                                           guint64                  count)
{
  FoundryDapDebuggerVariable *self = (FoundryDapDebuggerVariable *)variable;
  g_autoptr(FoundryDapDebugger) debugger = NULL;
  const char *memory_reference = NULL;
  DexFuture *future;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER_VARIABLE (self));
  g_assert (count > 0);

  if (!(debugger = g_weak_ref_get (&self->debugger_wr)))
    return foundry_future_new_disposed ();

  if (!FOUNDRY_JSON_OBJECT_PARSE (self->node,
                                  "memoryReference", FOUNDRY_JSON_NODE_GET_STRING (&memory_reference)))
    return foundry_future_new_not_supported ();

  future = foundry_dap_debugger_call (debugger,
                                      FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                               "command", "readMemory",
                                                               "arguments", "{",
                                                                 "memoryReference", FOUNDRY_JSON_NODE_PUT_STRING (memory_reference),
                                                                 "offset", FOUNDRY_JSON_NODE_PUT_INT (MIN (offset, G_MAXINT64)),
                                                                 "count", FOUNDRY_JSON_NODE_PUT_INT (MIN (count, G_MAXINT64)),
                                                               "}"));
  future = dex_future_then (future,
                            foundry_dap_protocol_unwrap_error,
                            NULL, NULL);
  future = dex_future_then (future,
                            foundry_dap_debugger_variable_read_memory_cb,
                            g_object_ref (self),
                            g_object_unref);
  return future;
}

static void
foundry_dap_debugger_variable_finalize (GObject *object)
{
  FoundryDapDebuggerVariable *self = (FoundryDapDebuggerVariable *)object;

  g_clear_pointer (&self->node, json_node_unref);
  g_weak_ref_clear (&self->debugger_wr);

  G_OBJECT_CLASS (foundry_dap_debugger_variable_parent_class)->finalize (object);
}

static void
foundry_dap_debugger_variable_class_init (FoundryDapDebuggerVariableClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDebuggerVariableClass *variable_Class = FOUNDRY_DEBUGGER_VARIABLE_CLASS (klass);

  object_class->finalize = foundry_dap_debugger_variable_finalize;

  variable_Class->dup_name = foundry_dap_debugger_variable_dup_name;
  variable_Class->dup_type_name = foundry_dap_debugger_variable_dup_type_name;
  variable_Class->dup_value = foundry_dap_debugger_variable_dup_value;
  variable_Class->is_structured = foundry_dap_debugger_variable_is_structured;
  variable_Class->list_children = foundry_dap_debugger_variable_list_children;
  variable_Class->read_memory = foundry_dap_debugger_variable_read_memory;
}

static void
foundry_dap_debugger_variable_init (FoundryDapDebuggerVariable *self)
{
  g_weak_ref_init (&self->debugger_wr, NULL);
}

FoundryDebuggerVariable *
foundry_dap_debugger_variable_new (FoundryDapDebugger *debugger,
                                   JsonNode           *node)
{
  FoundryDapDebuggerVariable *self;

  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER (debugger), NULL);
  g_return_val_if_fail (node != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_DAP_DEBUGGER_VARIABLE, NULL);
  g_weak_ref_set (&self->debugger_wr, debugger);
  self->node = json_node_ref (node);

  return FOUNDRY_DEBUGGER_VARIABLE (self);
}
