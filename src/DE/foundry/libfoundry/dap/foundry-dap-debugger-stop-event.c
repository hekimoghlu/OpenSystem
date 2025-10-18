/* foundry-dap-debugger-stop-event.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "foundry-debugger-trap.h"
#include "foundry-dap-debugger-stop-event-private.h"
#include "foundry-json-node.h"

struct _FoundryDapDebuggerStopEvent
{
  FoundryDebuggerStopEvent  parent_instance;
  FoundryDapDebugger       *debugger;
  JsonNode                 *node;
  GListStore               *traps;
};

G_DEFINE_FINAL_TYPE (FoundryDapDebuggerStopEvent, foundry_dap_debugger_stop_event, FOUNDRY_TYPE_DEBUGGER_STOP_EVENT)

static FoundryDebuggerStopReason
foundry_dap_debugger_stop_event_get_reason (FoundryDebuggerStopEvent *event)
{
  return 0;
}

static FoundryDebuggerTrap *
foundry_dap_debugger_stop_event_dup_trap (FoundryDebuggerStopEvent *event)
{
  FoundryDapDebuggerStopEvent *self = FOUNDRY_DAP_DEBUGGER_STOP_EVENT (event);

  if (g_list_model_get_n_items (G_LIST_MODEL (self->traps)) == 0)
    return NULL;

  return g_list_model_get_item (G_LIST_MODEL (self->traps), 0);
}

static void
foundry_dap_debugger_stop_event_finalize (GObject *object)
{
  FoundryDapDebuggerStopEvent *self = (FoundryDapDebuggerStopEvent *)object;

  g_clear_object (&self->debugger);
  g_clear_object (&self->traps);
  g_clear_pointer (&self->node, json_node_unref);

  G_OBJECT_CLASS (foundry_dap_debugger_stop_event_parent_class)->finalize (object);
}

static void
foundry_dap_debugger_stop_event_class_init (FoundryDapDebuggerStopEventClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDebuggerStopEventClass *stop_event_class = FOUNDRY_DEBUGGER_STOP_EVENT_CLASS (klass);

  object_class->finalize = foundry_dap_debugger_stop_event_finalize;

  stop_event_class->get_reason = foundry_dap_debugger_stop_event_get_reason;
  stop_event_class->dup_trap = foundry_dap_debugger_stop_event_dup_trap;
}

static void
foundry_dap_debugger_stop_event_init (FoundryDapDebuggerStopEvent *self)
{
  self->traps = g_list_store_new (FOUNDRY_TYPE_DEBUGGER_TRAP);
}

FoundryDebuggerEvent *
foundry_dap_debugger_stop_event_new (FoundryDapDebugger *debugger,
                                     JsonNode           *node)
{
  g_autoptr(FoundryDapDebuggerStopEvent) self = NULL;
  g_autoptr(GListModel) traps = NULL;
  const char *reason = NULL;
  const char *description = NULL;
  const char *text = NULL;
  JsonNode *body = NULL;
  JsonNode *breakpoint_ids = NULL;
  gboolean all_threads_stopped = FALSE;
  gint64 thread_id;

  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER (debugger), NULL);
  g_return_val_if_fail (node != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_DAP_DEBUGGER_STOP_EVENT, NULL);
  self->debugger = g_object_ref (debugger);
  self->node = json_node_ref (node);

  traps = foundry_debugger_list_traps (FOUNDRY_DEBUGGER (debugger));

  if (!FOUNDRY_JSON_OBJECT_PARSE (node, "body", FOUNDRY_JSON_NODE_GET_NODE (&body)))
    return NULL;

  if (!FOUNDRY_JSON_OBJECT_PARSE (body, "reason", FOUNDRY_JSON_NODE_GET_STRING (&reason)))
    return NULL;

  FOUNDRY_JSON_OBJECT_PARSE (body, "description", FOUNDRY_JSON_NODE_GET_STRING (&description));
  FOUNDRY_JSON_OBJECT_PARSE (body, "text", FOUNDRY_JSON_NODE_GET_STRING (&text));
  FOUNDRY_JSON_OBJECT_PARSE (body, "threadId", FOUNDRY_JSON_NODE_GET_INT (&thread_id));
  FOUNDRY_JSON_OBJECT_PARSE (body, "allThreadsStopped", FOUNDRY_JSON_NODE_GET_BOOLEAN (&all_threads_stopped));

  if (traps != NULL &&
      FOUNDRY_JSON_OBJECT_PARSE (body, "hitBreakpointIds", FOUNDRY_JSON_NODE_GET_NODE (&breakpoint_ids)) &&
      JSON_NODE_HOLDS_ARRAY (breakpoint_ids))
    {
      JsonArray *ar = json_node_get_array (breakpoint_ids);
      guint length = json_array_get_length (ar);
      guint n_items = g_list_model_get_n_items (traps);

      for (guint i = 0; i < length; i++)
        {
          gint64 breakpoint_id = json_array_get_int_element (ar, i);
          g_autofree char *id_str = NULL;

          if (breakpoint_id <= 0)
            continue;

          id_str = g_strdup_printf ("%"G_GINT64_FORMAT, breakpoint_id);

          for (guint j = 0; j < n_items; j++)
            {
              g_autoptr(FoundryDebuggerTrap) trap = g_list_model_get_item (traps, j);
              g_autofree char *id = foundry_debugger_trap_dup_id (trap);

              if (g_strcmp0 (id, id_str) == 0)
                {
                  g_list_store_append (self->traps, trap);
                  break;
                }
            }
        }
    }

  return FOUNDRY_DEBUGGER_EVENT (g_steal_pointer (&self));
}
