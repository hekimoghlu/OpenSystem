/* foundry-dap-debugger.c
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

#include "foundry-command.h"
#include "foundry-dap-debugger-private.h"
#include "foundry-dap-debugger-breakpoint-private.h"
#include "foundry-dap-debugger-instruction-private.h"
#include "foundry-dap-debugger-log-message-private.h"
#include "foundry-dap-debugger-module-private.h"
#include "foundry-dap-debugger-stop-event-private.h"
#include "foundry-dap-debugger-thread-private.h"
#include "foundry-dap-debugger-watchpoint-private.h"
#include "foundry-dap-driver-private.h"
#include "foundry-dap-protocol.h"
#include "foundry-debugger-target.h"
#include "foundry-debugger-target-command.h"
#include "foundry-debugger-target-process.h"
#include "foundry-debugger-trap-params.h"
#include "foundry-json-node.h"
#include "foundry-util-private.h"

typedef struct
{
  GIOStream               *stream;
  GSubprocess             *subprocess;
  FoundryDapDriver        *driver;
  GListStore              *log_messages;
  GListStore              *modules;
  GListStore              *threads;
  GListStore              *traps;
  GPtrArray               *trap_params;
  DexPromise              *sync_params;
  guint                    sync_params_source;
  FoundryDapDebuggerQuirk  quirks;
  FoundryDebuggerThread   *primary_thread;
  guint                    has_terminated : 1;
} FoundryDapDebuggerPrivate;

enum {
  PROP_0,
  PROP_QUIRKS,
  PROP_STREAM,
  PROP_SUBPROCESS,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryDapDebugger, foundry_dap_debugger, FOUNDRY_TYPE_DEBUGGER)

static GParamSpec *properties[N_PROPS];

static inline DexFuture *
foundry_dap_debugger_call_checked (FoundryDapDebugger *self,
                                   JsonNode           *node)
{
  return dex_future_then (foundry_dap_debugger_call (self, node),
                          foundry_dap_protocol_unwrap_error,
                          NULL, NULL);
}

static gint64
get_default_thread_id (FoundryDapDebugger *self)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));

  if (g_list_model_get_n_items (G_LIST_MODEL (priv->threads)) > 0)
    {
      g_autoptr(FoundryDapDebuggerThread) thread = g_list_model_get_item (G_LIST_MODEL (priv->threads), 0);

      return foundry_dap_debugger_thread_get_id (thread);
    }

  return 1;
}

static DexFuture *
foundry_dap_debugger_query_threads_cb (DexFuture *completed,
                                       gpointer   user_data)
{
  FoundryDapDebugger *self = user_data;
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);
  g_autoptr(GPtrArray) all_threads = NULL;
  g_autoptr(JsonNode) reply = NULL;
  JsonNode *threads = NULL;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));

  if ((priv->quirks & FOUNDRY_DAP_DEBUGGER_QUIRK_QUERY_THREADS) == 0)
    return dex_ref (completed);

  all_threads = g_ptr_array_new_with_free_func (g_object_unref);

  if ((reply = dex_await_boxed (dex_ref (completed), NULL)) &&
      FOUNDRY_JSON_OBJECT_PARSE (reply,
                                 "body", "{",
                                   "threads", FOUNDRY_JSON_NODE_GET_NODE (&threads),
                                 "}") &&
      JSON_NODE_HOLDS_ARRAY (threads))
    {
      JsonArray *ar = json_node_get_array (threads);
      guint length = json_array_get_length (ar);

      for (guint i = 0; i < length; i++)
        {
          JsonNode *element = json_array_get_element (ar, i);
          const char *name = NULL;
          gint64 thread_id = 0;

          if (FOUNDRY_JSON_OBJECT_PARSE (element,
                                         "id", FOUNDRY_JSON_NODE_GET_INT (&thread_id),
                                         "name", FOUNDRY_JSON_NODE_GET_STRING (&name)))
            {
              g_autoptr(FoundryDebuggerThread) thread = NULL;

              if ((thread = foundry_dap_debugger_thread_new (self, thread_id)))
                g_ptr_array_add (all_threads, g_steal_pointer (&thread));
            }
        }
    }

  g_list_store_splice (priv->threads,
                       0,
                       g_list_model_get_n_items (G_LIST_MODEL (priv->threads)),
                       all_threads->pdata,
                       all_threads->len);

  return dex_ref (completed);
}

static void
foundry_dap_debugger_query_threads (FoundryDapDebugger *self)
{
  DexFuture *future;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));

  future = foundry_dap_debugger_call (self,
                                      FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                               "command", "threads",
                                                               "arguments", "{", "}"));
  future = dex_future_then (future,
                            foundry_dap_protocol_unwrap_error,
                            NULL, NULL);
  future = dex_future_then (future,
                            foundry_dap_debugger_query_threads_cb,
                            g_object_ref (self),
                            g_object_unref);
  dex_future_disown (future);
}

static void
foundry_dap_debugger_handle_output_event (FoundryDapDebugger *self,
                                          JsonNode           *node)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);
  g_autoptr(FoundryDebuggerLogMessage) message = NULL;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));
  g_assert (node != NULL);

  if ((message = foundry_dap_debugger_log_message_new (node)))
    g_list_store_append (priv->log_messages, message);
}

static void
foundry_dap_debugger_handle_module_event (FoundryDapDebugger *self,
                                          JsonNode           *node)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);
  const char *reason = NULL;
  const char *id = NULL;
  const char *name = NULL;
  const char *path = NULL;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));
  g_assert (node != NULL);

  if (!FOUNDRY_JSON_OBJECT_PARSE (node,
                                  "body", "{",
                                    "reason", FOUNDRY_JSON_NODE_GET_STRING (&reason),
                                    "module", "{",
                                      "id", FOUNDRY_JSON_NODE_GET_STRING (&id),
                                      "name", FOUNDRY_JSON_NODE_GET_STRING (&name),
                                      "path", FOUNDRY_JSON_NODE_GET_STRING (&path),
                                    "}",
                                  "}"))
    return;

  if (!FOUNDRY_JSON_OBJECT_PARSE (node,
                                  "body", "{",
                                    "module", "{",
                                      "path", FOUNDRY_JSON_NODE_GET_STRING (&path),
                                    "}",
                                  "}"))
    path = NULL;

  if (g_strcmp0 (reason, "changed") == 0 ||
      g_strcmp0 (reason, "removed") == 0)
    {
      guint n_items = g_list_model_get_n_items (G_LIST_MODEL (priv->modules));

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryDebuggerModule) module = g_list_model_get_item (G_LIST_MODEL (priv->modules), i);
          g_autofree char *module_id = foundry_debugger_module_dup_id (module);

          if (g_strcmp0 (id, module_id) == 0)
            {
              g_list_store_remove (priv->modules, i);
              break;
            }
        }
    }

  if (g_strcmp0 (reason, "new") == 0 ||
      g_strcmp0 (reason, "changed") == 0)
    {
      g_autoptr(FoundryDebuggerModule) module = NULL;

      module = foundry_dap_debugger_module_new (self, id, name, path);
      g_list_store_append (priv->modules, module);
    }
}

static void
mark_thread_stopped (GListModel *threads,
                     gint64      thread_id,
                     gboolean    stopped)
{
  guint n_items = g_list_model_get_n_items (threads);
  g_autofree char *id_str = g_strdup_printf ("%"G_GINT64_FORMAT, thread_id);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDapDebuggerThread) thread = g_list_model_get_item (threads, i);
      g_autofree char *id = foundry_debugger_thread_dup_id (FOUNDRY_DEBUGGER_THREAD (thread));

      if (thread_id == 0 || foundry_str_equal0 (id, id_str))
        foundry_dap_debugger_thread_set_stopped (thread, stopped);
    }
}

static void
foundry_dap_debugger_handle_stopped_event (FoundryDapDebugger *self,
                                           JsonNode           *node)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);
  g_autoptr(FoundryDebuggerEvent) event = NULL;
  JsonNode *body = NULL;
  const char *reason = NULL;
  gint64 thread_id = 0;
  gboolean all_threads_stopped = FALSE;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));
  g_assert (node != NULL);

  if (!FOUNDRY_JSON_OBJECT_PARSE (node, "body", FOUNDRY_JSON_NODE_GET_NODE (&body)))
    return;

  if (!FOUNDRY_JSON_OBJECT_PARSE (body, "reason", FOUNDRY_JSON_NODE_GET_STRING (&reason)))
    return;

  if (!FOUNDRY_JSON_OBJECT_PARSE (body, "threadId", FOUNDRY_JSON_NODE_GET_INT (&thread_id)))
    thread_id = 0;

  if (FOUNDRY_JSON_OBJECT_PARSE (body, "allThreadsStopped", FOUNDRY_JSON_NODE_GET_BOOLEAN (&all_threads_stopped)) &&
      all_threads_stopped)
    mark_thread_stopped (G_LIST_MODEL (priv->threads), 0, TRUE);
  else
    mark_thread_stopped (G_LIST_MODEL (priv->threads), thread_id, TRUE);

  event = foundry_dap_debugger_stop_event_new (self, node);
  foundry_debugger_emit_event (FOUNDRY_DEBUGGER (self), event);

  if ((priv->quirks & FOUNDRY_DAP_DEBUGGER_QUIRK_QUERY_THREADS) != 0)
    foundry_dap_debugger_query_threads (self);
}

static void
foundry_dap_debugger_handle_terminated_event (FoundryDapDebugger *self,
                                              JsonNode           *node)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));
  g_assert (node != NULL);

  priv->has_terminated = TRUE;

  g_object_notify (G_OBJECT (self), "terminated");
}

static void
foundry_dap_debugger_handle_continued_event (FoundryDapDebugger *self,
                                             JsonNode           *node)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);
  JsonNode *body = NULL;
  gint64 thread_id = 0;
  gboolean all_threads_continued = FALSE;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));
  g_assert (node != NULL);

  if (!FOUNDRY_JSON_OBJECT_PARSE (node, "body", FOUNDRY_JSON_NODE_GET_NODE (&body)))
    return;

  if (!FOUNDRY_JSON_OBJECT_PARSE (body, "threadId", FOUNDRY_JSON_NODE_GET_INT (&thread_id)))
    thread_id = 0;

  if (FOUNDRY_JSON_OBJECT_PARSE (body, "allThreadsContinued", FOUNDRY_JSON_NODE_GET_BOOLEAN (&all_threads_continued)) &&
      all_threads_continued)
    mark_thread_stopped (G_LIST_MODEL (priv->threads), 0, FALSE);
  else
    mark_thread_stopped (G_LIST_MODEL (priv->threads), thread_id, FALSE);
}

static void
foundry_dap_debugger_handle_thread_event (FoundryDapDebugger *self,
                                          JsonNode           *node)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);
  const char *reason = NULL;
  gint64 thread_id = 0;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));
  g_assert (node != NULL);

  if (!FOUNDRY_JSON_OBJECT_PARSE (node,
                                  "body", "{",
                                    "reason", FOUNDRY_JSON_NODE_GET_STRING (&reason),
                                    "threadId", FOUNDRY_JSON_NODE_GET_INT (&thread_id),
                                  "}"))
    return;

  if (foundry_str_equal0 (reason, "started"))
    {
      g_autoptr(FoundryDebuggerThread) thread = NULL;

      if ((thread = foundry_dap_debugger_thread_new (self, thread_id)))
        {
          g_list_store_append (priv->threads, thread);

          /* Set the first thread as the primary thread */
          if (priv->primary_thread == NULL)
            {
              priv->primary_thread = g_object_ref (thread);
              g_object_notify (G_OBJECT (self), "primary-thread");
            }
        }
    }
  else if (foundry_str_equal0 (reason, "exited"))
    {
      g_autofree char *id_str = g_strdup_printf ("%"G_GINT64_FORMAT, thread_id);
      guint n_items = g_list_model_get_n_items (G_LIST_MODEL (priv->threads));

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryDebuggerThread) thread = g_list_model_get_item (G_LIST_MODEL (priv->threads), i);
          g_autofree char *id = foundry_debugger_thread_dup_id (thread);

          if (foundry_str_equal0 (id, id_str))
            {
              g_list_store_remove (priv->threads, i);
              break;
            }
        }
    }
}

static void
foundry_dap_debugger_handle_breakpoint_event (FoundryDapDebugger *self,
                                              JsonNode           *node)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);
  JsonNode *body = NULL;
  JsonNode *breakpoint = NULL;
  const char *reason = NULL;
  gint64 breakpoint_id = 0;
  guint n_items;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));
  g_assert (node != NULL);

  if (!FOUNDRY_JSON_OBJECT_PARSE (node, "body", FOUNDRY_JSON_NODE_GET_NODE (&body)))
    return;

  if (!FOUNDRY_JSON_OBJECT_PARSE (body,
                                  "reason", FOUNDRY_JSON_NODE_GET_STRING (&reason),
                                  "breakpoint", FOUNDRY_JSON_NODE_GET_NODE (&breakpoint)))
    return;

  if (!FOUNDRY_JSON_OBJECT_PARSE (breakpoint, "id", FOUNDRY_JSON_NODE_GET_INT (&breakpoint_id)))
    return;

  /* Find existing trap and update it */
  n_items = g_list_model_get_n_items (G_LIST_MODEL (priv->traps));
  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDebuggerTrap) trap = g_list_model_get_item (G_LIST_MODEL (priv->traps), i);
      g_autofree char *trap_id = foundry_debugger_trap_dup_id (trap);
      g_autofree char *id_str = g_strdup_printf ("%"G_GINT64_FORMAT, breakpoint_id);

      if (foundry_str_equal0 (trap_id, id_str))
        {
          /* Update the existing trap */
          if (g_strcmp0 (reason, "changed") == 0)
            {
              /* Replace the trap with updated information */
              g_autoptr(FoundryDebuggerTrap) new_trap = NULL;

              if (FOUNDRY_IS_DAP_DEBUGGER_BREAKPOINT (trap))
                new_trap = FOUNDRY_DEBUGGER_TRAP (foundry_dap_debugger_breakpoint_new (self, breakpoint));
              else if (FOUNDRY_IS_DAP_DEBUGGER_WATCHPOINT (trap))
                new_trap = FOUNDRY_DEBUGGER_TRAP (foundry_dap_debugger_watchpoint_new (self, breakpoint));

              if (new_trap != NULL)
                {
                  g_list_store_remove (priv->traps, i);
                  g_list_store_insert (priv->traps, i, new_trap);
                }
            }
          else if (g_strcmp0 (reason, "removed") == 0)
            {
              g_list_store_remove (priv->traps, i);
            }
          break;
        }
    }

  /* If we didn't find an existing trap and this is a "new" event, add it */
  if (g_strcmp0 (reason, "new") == 0)
    {
      g_autoptr(FoundryDebuggerTrap) new_trap = NULL;
      const char *instruction = NULL;
      gint64 line;

      /* Determine trap type based on breakpoint properties */
      if (FOUNDRY_JSON_OBJECT_PARSE (breakpoint, "line", FOUNDRY_JSON_NODE_GET_INT (&line)))
        {
          /* This is a source breakpoint */
          new_trap = FOUNDRY_DEBUGGER_TRAP (foundry_dap_debugger_breakpoint_new (self, breakpoint));
        }
      else if (FOUNDRY_JSON_OBJECT_PARSE (breakpoint, "instructionReference", FOUNDRY_JSON_NODE_GET_STRING (&instruction)))
        {
          /* This is an instruction breakpoint - treat as breakpoint for now */
          new_trap = FOUNDRY_DEBUGGER_TRAP (foundry_dap_debugger_breakpoint_new (self, breakpoint));
        }
      else
        {
          /* This might be a data breakpoint (watchpoint) */
          new_trap = FOUNDRY_DEBUGGER_TRAP (foundry_dap_debugger_watchpoint_new (self, breakpoint));
        }

      if (new_trap != NULL)
        g_list_store_append (priv->traps, new_trap);
    }
}

static void
foundry_dap_debugger_handle_initialized (FoundryDapDebugger *self,
                                         JsonNode           *node)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));
  g_assert (node != NULL);

  if ((priv->quirks & FOUNDRY_DAP_DEBUGGER_QUIRK_QUERY_THREADS) != 0)
    foundry_dap_debugger_query_threads (self);
}

static void
foundry_dap_debugger_driver_event_cb (FoundryDapDebugger *self,
                                      JsonNode           *node,
                                      FoundryDapDriver   *driver)
{
  const char *event = NULL;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));
  g_assert (node != NULL);
  g_assert (FOUNDRY_IS_DAP_DRIVER (driver));

  if (!FOUNDRY_JSON_OBJECT_PARSE (node,
                                  "type", "event",
                                  "event", FOUNDRY_JSON_NODE_GET_STRING (&event)))
    return;

  if (FALSE) {}
  else if (g_strcmp0 (event, "output") == 0)
    foundry_dap_debugger_handle_output_event (self, node);
  else if (g_strcmp0 (event, "module") == 0)
    foundry_dap_debugger_handle_module_event (self, node);
  else if (g_strcmp0 (event, "stopped") == 0)
    foundry_dap_debugger_handle_stopped_event (self, node);
  else if (g_strcmp0 (event, "thread") == 0)
    foundry_dap_debugger_handle_thread_event (self, node);
  else if (g_strcmp0 (event, "continued") == 0)
    foundry_dap_debugger_handle_continued_event (self, node);
  else if (g_strcmp0 (event, "initialized") == 0)
    foundry_dap_debugger_handle_initialized (self, node);
  else if (g_strcmp0 (event, "breakpoint") == 0)
    foundry_dap_debugger_handle_breakpoint_event (self, node);
  else if (g_strcmp0 (event, "terminated") == 0)
    foundry_dap_debugger_handle_terminated_event (self, node);
}

static gboolean
foundry_dap_debugger_driver_handle_request_cb (FoundryDapDebugger *self,
                                               JsonNode           *node,
                                               FoundryDapDriver   *driver)
{
  const char *command = NULL;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));
  g_assert (node != NULL);
  g_assert (FOUNDRY_IS_DAP_DRIVER (driver));

  if (FOUNDRY_JSON_OBJECT_PARSE (node, "command", FOUNDRY_JSON_NODE_GET_STRING (&command)))
    g_debug ("DAP requested method `%s`", command);

  return FALSE;
}

static DexFuture *
foundry_dap_debugger_exited (DexFuture *future,
                             gpointer   user_data)
{
  GWeakRef *wr = user_data;
  FoundryDapDebuggerPrivate *priv;
  g_autoptr(FoundryDapDebugger) self = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (DEX_IS_FUTURE (future));

  if (!(self = g_weak_ref_get (wr)))
    return dex_future_new_true ();

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));

  priv = foundry_dap_debugger_get_instance_private (self);

  if (!dex_await (dex_ref (future), &error))
    {
      if (priv->stream != NULL)
        g_io_stream_close (priv->stream, NULL, NULL);
    }

  return dex_ref (future);
}

static GListModel *
foundry_dap_debugger_list_log_messages (FoundryDebugger *debugger)
{
  FoundryDapDebugger *self = FOUNDRY_DAP_DEBUGGER (debugger);
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  return g_object_ref (G_LIST_MODEL (priv->log_messages));
}

static GListModel *
foundry_dap_debugger_list_modules (FoundryDebugger *debugger)
{
  FoundryDapDebugger *self = FOUNDRY_DAP_DEBUGGER (debugger);
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  return g_object_ref (G_LIST_MODEL (priv->modules));
}

static GListModel *
foundry_dap_debugger_list_threads (FoundryDebugger *debugger)
{
  FoundryDapDebugger *self = FOUNDRY_DAP_DEBUGGER (debugger);
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  return g_object_ref (G_LIST_MODEL (priv->threads));
}

static GListModel *
foundry_dap_debugger_list_traps (FoundryDebugger *debugger)
{
  FoundryDapDebugger *self = FOUNDRY_DAP_DEBUGGER (debugger);
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  return g_object_ref (G_LIST_MODEL (priv->traps));
}

static FoundryDebuggerThread *
foundry_dap_debugger_dup_primary_thread (FoundryDebugger *debugger)
{
  FoundryDapDebugger *self = FOUNDRY_DAP_DEBUGGER (debugger);
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  if (priv->primary_thread)
    return g_object_ref (priv->primary_thread);

  return NULL;
}

static DexFuture *
continue_success_cb (DexFuture *completed,
                     gpointer   user_data)
{
  g_autoptr(JsonNode) node = dex_await_boxed (dex_ref (completed), NULL);
  FoundryDapDebugger *self = user_data;
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);
  gboolean all = FALSE;

  if (FOUNDRY_JSON_OBJECT_PARSE (node,
                                 "type", "response",
                                 "command", "continue",
                                 "body", "{",
                                   "allThreadsContinued", FOUNDRY_JSON_NODE_GET_BOOLEAN (&all),
                                 "}") && all)
    mark_thread_stopped (G_LIST_MODEL (priv->threads), 0, FALSE);

  return dex_ref (completed);
}

DexFuture *
_foundry_dap_debugger_move (FoundryDapDebugger      *self,
                            gint64                   thread_id,
                            FoundryDebuggerMovement  movement)
{
  DexFuture *move = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_DAP_DEBUGGER (self));

  g_debug ("`%s` advancing thread %"G_GINT64_FORMAT" with movement 0x%x",
           G_OBJECT_TYPE_NAME (self), thread_id, movement);

  switch (movement)
    {
    case FOUNDRY_DEBUGGER_MOVEMENT_START:
      /* We "start" automatically, fallthrough to continue */
      G_GNUC_FALLTHROUGH;

    case FOUNDRY_DEBUGGER_MOVEMENT_CONTINUE:
      move = foundry_dap_debugger_call (self,
                                        FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                 "command", "continue",
                                                                 "arguments", "{",
                                                                   "threadId", FOUNDRY_JSON_NODE_PUT_INT (thread_id),
                                                                 "}"));
      move = dex_future_then (move,
                              continue_success_cb,
                              g_object_ref (self),
                              g_object_unref);
      break;

    case FOUNDRY_DEBUGGER_MOVEMENT_STEP_IN:
      move = foundry_dap_debugger_call (self,
                                        FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                 "command", "stepIn",
                                                                 "arguments", "{",
                                                                   "threadId", FOUNDRY_JSON_NODE_PUT_INT (thread_id),
                                                                 "}"));
      break;

    case FOUNDRY_DEBUGGER_MOVEMENT_STEP_OVER:
      move = foundry_dap_debugger_call (self,
                                        FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                 "command", "next",
                                                                 "arguments", "{",
                                                                   "threadId", FOUNDRY_JSON_NODE_PUT_INT (thread_id),
                                                                 "}"));
      break;

    case FOUNDRY_DEBUGGER_MOVEMENT_STEP_OUT:
      move = foundry_dap_debugger_call (self,
                                        FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                 "command", "stepOut",
                                                                 "arguments", "{",
                                                                   "threadId", FOUNDRY_JSON_NODE_PUT_INT (thread_id),
                                                                 "}"));
      break;

    default:
      g_assert_not_reached ();
    }

  if (move != NULL)
    move = dex_future_then (move,
                            foundry_dap_protocol_unwrap_error,
                            NULL, NULL);

  return g_steal_pointer (&move);
}

static DexFuture *
foundry_dap_debugger_move (FoundryDebugger         *debugger,
                           FoundryDebuggerMovement  movement)
{
  FoundryDapDebugger *self = FOUNDRY_DAP_DEBUGGER (debugger);
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);
  gint64 id = 1;

  if (g_list_model_get_n_items (G_LIST_MODEL (priv->threads)) != 0)
    {
      g_autoptr(FoundryDebuggerThread) thread = NULL;
      g_autofree char *thread_id = NULL;

      thread = g_list_model_get_item (G_LIST_MODEL (priv->threads), 0);
      thread_id = foundry_debugger_thread_dup_id (thread);
      id = g_ascii_strtoll (thread_id, NULL, 10);
    }

  return _foundry_dap_debugger_move (FOUNDRY_DAP_DEBUGGER (debugger), id, movement);
}

static DexFuture *
foundry_dap_debugger_interrupt (FoundryDebugger *debugger)
{
  FoundryDapDebugger *self = FOUNDRY_DAP_DEBUGGER (debugger);
  gint64 thread_id = get_default_thread_id (self);

  return foundry_dap_debugger_call_checked (self,
                                            FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                     "command", "pause",
                                                                     "arguments", "{",
                                                                       "threadId", FOUNDRY_JSON_NODE_PUT_INT (thread_id),
                                                                     "}"));
}

static DexFuture *
foundry_dap_debugger_interpret (FoundryDebugger *debugger,
                                const char      *text)
{
  return foundry_dap_debugger_call_checked (FOUNDRY_DAP_DEBUGGER (debugger),
                                            FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                     "command", "evaluate",
                                                                     "arguments", "{",
                                                                       "context", "repl",
                                                                       "expression", FOUNDRY_JSON_NODE_PUT_STRING (text),
                                                                     "}"));
}

static JsonNode *
create_function_node (FoundryDebuggerTrapParams *params)
{
  g_autofree char *function = foundry_debugger_trap_params_dup_function (params);

  return FOUNDRY_JSON_OBJECT_NEW ("name", function);
}

static JsonNode *
create_instruction_node (FoundryDebuggerTrapParams *params)
{
  guint64 ip = foundry_debugger_trap_params_get_instruction_pointer (params);
  g_autofree char *ip_str = g_strdup_printf ("0x%"G_GINT64_MODIFIER"x", ip);

  return FOUNDRY_JSON_OBJECT_NEW ("instructionReference", ip_str);
}

static JsonNode *
create_breakpoint_node (FoundryDebuggerTrapParams *params)
{
  guint line = foundry_debugger_trap_params_get_line (params);
  guint line_offset = foundry_debugger_trap_params_get_line_offset (params);

  return FOUNDRY_JSON_OBJECT_NEW ("line", line,
                                  "column", line_offset);
}

static JsonNode *
create_array (void)
{
  JsonNode *node = json_node_new (JSON_NODE_ARRAY);
  g_autoptr(JsonArray) ar = json_array_new ();
  json_node_set_array (node, ar);
  return node;
}

static DexFuture *
foundry_dap_debugger_sync_traps_fiber (gpointer user_data)
{
  FoundryDapDebugger *self = user_data;
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);
  g_autoptr(GHashTable) by_path = NULL;
  g_autoptr(GPtrArray) functions = NULL;
  g_autoptr(GPtrArray) instructions = NULL;
  g_autoptr(GPtrArray) futures = NULL;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));

  futures = g_ptr_array_new_with_free_func (dex_unref);
  functions = g_ptr_array_new_with_free_func (g_object_unref);
  instructions = g_ptr_array_new_with_free_func (g_object_unref);
  by_path = g_hash_table_new_full ((GHashFunc) g_str_hash,
                                   (GEqualFunc) g_str_equal,
                                   g_free,
                                   (GDestroyNotify) g_ptr_array_unref);

  /* TODO:
   *
   * DAP does not have the concept of countpoints from what I can tell
   * when reading the spec. So from what I can tell we will have to
   * manually restart after incrementing a count on the breakpoint.
   *
   * What is tricky here is that we will not want to do expensive
   * quirks during those stop operations (such as issuing `modules`)
   * to update our modules list.
   *
   * Though even having to round-trip from the debugger to implement
   * this is already going to be too slow for some situations. We may
   * want to intruduce a way for subclasses to create the countpoint
   * for us so that we can use vendor-specific API (e.g. GDB evaluate).
   */

  for (guint i = 0; i < priv->trap_params->len; i++)
    {
      FoundryDebuggerTrapParams *params = g_ptr_array_index (priv->trap_params, i);
      g_autofree char *function = foundry_debugger_trap_params_dup_function (params);
      g_autofree char *path = foundry_debugger_trap_params_dup_path (params);
      guint64 instruction_pointer = foundry_debugger_trap_params_get_instruction_pointer (params);

      if (function != NULL)
        {
          g_ptr_array_add (functions, g_object_ref (params));
          continue;
        }

      if (instruction_pointer != 0)
        {
          g_ptr_array_add (instructions, g_object_ref (params));
          continue;
        }

      if (path != NULL)
        {
          GPtrArray *ar;

          if (!(ar = g_hash_table_lookup (by_path, path)))
            {
              ar = g_ptr_array_new_with_free_func (g_object_unref);
              g_hash_table_replace (by_path, g_strdup (path), ar);
            }

          continue;
        }

      g_debug ("Incomplete trap params");
    }

  if (functions->len > 0)
    {
      g_autoptr(JsonNode) functions_node = create_array ();
      JsonArray *functions_ar = json_node_get_array (functions_node);

      for (guint i = 0; i < functions->len; i++)
        {
          FoundryDebuggerTrapParams *params = g_ptr_array_index (functions, i);
          g_autoptr(JsonNode) function_node = create_function_node (params);

          if (function_node != NULL)
            json_array_add_element (functions_ar, g_steal_pointer (&function_node));
        }

      g_ptr_array_add (futures,
                       foundry_dap_debugger_call_checked (self,
                                                          FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                                   "command", "setFunctionBreakpoints",
                                                                                   "arguments", "{",
                                                                                     "breakpoints", FOUNDRY_JSON_NODE_PUT_NODE (functions_node),
                                                                                   "}")));
    }

  if (instructions->len > 0)
    {
      g_autoptr(JsonNode) instructions_node = create_array ();
      JsonArray *instructions_ar = json_node_get_array (instructions_node);

      for (guint i = 0; i < instructions->len; i++)
        {
          FoundryDebuggerTrapParams *params = g_ptr_array_index (instructions, i);
          g_autoptr(JsonNode) instruction_node = create_instruction_node (params);

          if (instruction_node != NULL)
            json_array_add_element (instructions_ar, g_steal_pointer (&instruction_node));
        }

      g_ptr_array_add (futures,
                       foundry_dap_debugger_call_checked (self,
                                                          FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                                   "command", "setInstructionBreakpoints",
                                                                                   "arguments", "{",
                                                                                     "breakpoints", FOUNDRY_JSON_NODE_PUT_NODE (instructions_node),
                                                                                   "}")));
    }

  if (g_hash_table_size (by_path) > 0)
    {
      GHashTableIter iter;
      gpointer key, value;

      g_hash_table_iter_init (&iter, by_path);

      while (g_hash_table_iter_next (&iter, &key, &value))
        {
          g_autoptr(JsonNode) breakpoints_node = create_array ();
          JsonArray *breakpoints_ar = json_node_get_array (breakpoints_node);
          const char *path = key;
          GPtrArray *ar = value;

          for (guint i = 0; i < ar->len; i++)
            {
              FoundryDebuggerTrapParams *params = g_ptr_array_index (ar, i);
              g_autoptr(JsonNode) breakpoint_node = create_breakpoint_node (params);

              if (breakpoint_node != NULL)
                json_array_add_element (breakpoints_ar, g_steal_pointer (&breakpoint_node));
            }

          g_ptr_array_add (futures,
                           foundry_dap_debugger_call_checked (self,
                                                              FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                                       "command", "setBreakpoints",
                                                                                       "arguments", "{",
                                                                                         "source", "{",
                                                                                           "path", FOUNDRY_JSON_NODE_PUT_STRING (path),
                                                                                         "}",
                                                                                         "breakpoints", FOUNDRY_JSON_NODE_PUT_NODE (breakpoints_node),
                                                                                       "}")));

        }
    }

  if (futures->len > 0)
    dex_await (foundry_future_all (futures), NULL);

  /* TODO: Parse the responses and create trap objects for the traps list store */
  /* This would require parsing the setBreakpoints/setFunctionBreakpoints/setInstructionBreakpoints responses */

  return dex_future_new_true ();
}

static gboolean
foundry_dap_debugger_sync_traps (gpointer user_data)
{
  FoundryDapDebugger *self = user_data;
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);
  g_autoptr(DexPromise) promise = NULL;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));

  g_clear_handle_id (&priv->sync_params_source, g_source_remove);

  if ((promise = g_steal_pointer (&priv->sync_params)))
    {
      dex_future_disown (dex_scheduler_spawn (NULL, 0,
                                              foundry_dap_debugger_sync_traps_fiber,
                                              g_object_ref (self),
                                              g_object_unref));

      dex_promise_resolve_boolean (promise, TRUE);
    }

  return G_SOURCE_REMOVE;
}

static DexFuture *
foundry_dap_debugger_trap (FoundryDebugger           *debugger,
                           FoundryDebuggerTrapParams *params)
{
  FoundryDapDebugger *self = (FoundryDapDebugger *)debugger;
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));
  g_assert (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (params));

  g_ptr_array_add (priv->trap_params, foundry_debugger_trap_params_copy (params));

  if (priv->sync_params == NULL)
    {
      priv->sync_params = dex_promise_new ();
      priv->sync_params_source = g_idle_add (foundry_dap_debugger_sync_traps, self);
    }

  return dex_ref (DEX_FUTURE (priv->sync_params));
}

static gboolean
foundry_dap_debugger_can_move (FoundryDebugger         *debugger,
                               FoundryDebuggerMovement  movement)
{
  FoundryDapDebugger *self = (FoundryDapDebugger *)debugger;
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);
  g_autoptr(FoundryDebuggerThread) thread = NULL;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));

  if (movement == FOUNDRY_DEBUGGER_MOVEMENT_START)
    return FALSE;

  if (!(thread = g_list_model_get_item (G_LIST_MODEL (priv->threads), 0)))
    return FALSE;

  return foundry_debugger_thread_can_move (thread, movement);
}

static DexFuture *
foundry_dap_debugger_disassemble_cb (DexFuture *completed,
                                     gpointer   data)
{
  g_autoptr(GListStore) store = NULL;
  g_autoptr(JsonNode) node = NULL;
  JsonNode *instructions;

  g_assert (DEX_IS_FUTURE (completed));

  store = g_list_store_new (FOUNDRY_TYPE_DEBUGGER_INSTRUCTION);

  if ((node = dex_await_boxed (dex_ref (completed), NULL)) &&
      FOUNDRY_JSON_OBJECT_PARSE (node,
                                 "body", "{",
                                   "instructions", FOUNDRY_JSON_NODE_GET_NODE (&instructions),
                                 "}") &&
      JSON_NODE_HOLDS_ARRAY (instructions))
    {
      JsonArray *ar = json_node_get_array (instructions);
      guint length = json_array_get_length (ar);

      for (guint i = 0; i < length; i++)
        {
          JsonNode *child = json_array_get_element (ar, i);
          g_autoptr(FoundryDebuggerInstruction) instruction = NULL;

          if ((instruction = foundry_dap_debugger_instruction_new (child)))
            g_list_store_append (store, instruction);
        }
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static DexFuture *
foundry_dap_debugger_disassemble (FoundryDebugger *debugger,
                                  guint64          begin_address,
                                  guint64          end_address)
{
  FoundryDapDebugger *self = (FoundryDapDebugger *)debugger;
  g_autofree char *begin_str = NULL;
  DexFuture *future;
  gint64 count;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER (self));

  if (end_address < begin_address)
    {
      guint64 tmp = begin_address;
      begin_address = end_address;
      end_address = tmp;
    }

  /* DAP Wants "number of instructions", not range. We of course don't
   * know that here so we just approximate it assuming that almost all
   * instructions are at least 2 bytes. It averages out to closer to
   * 3-5 on Intel and 2.5-3 on arm.
   */
  count = (end_address - begin_address) / 2;

  /* Assume 0xADDRESS memory references. That is likely the case though not
   * necessary guaranteed. To support it another way we'll have to go through a
   * memoryReference directly (such as a FoundryDebuggerVariable).
   */
  begin_str = g_strdup_printf ("0x%"G_GINT64_MODIFIER"x", begin_address);

  future = foundry_dap_debugger_call_checked (self,
                                              FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                       "command", "disassemble",
                                                                       "arguments", "{",
                                                                         "memoryReference", FOUNDRY_JSON_NODE_PUT_STRING (begin_str),
                                                                         "instructionCount", FOUNDRY_JSON_NODE_PUT_INT (count),
                                                                         "resolveSymbols", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE),
                                                                       "}"));
  future = dex_future_then (future,
                            foundry_dap_debugger_disassemble_cb,
                            NULL, NULL);

  return future;
}

static DexFuture *
foundry_dap_debugger_stop (FoundryDebugger *debugger)
{
  return foundry_dap_debugger_call_checked (FOUNDRY_DAP_DEBUGGER (debugger),
                                            FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                     "command", "terminate",
                                                                     "arguments", "{", "}"));
}

static gboolean
foundry_dap_debugger_has_terminated (FoundryDebugger *debugger)
{
  FoundryDapDebugger *self = FOUNDRY_DAP_DEBUGGER (debugger);
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  return priv->has_terminated;
}

static void
foundry_dap_debugger_constructed (GObject *object)
{
  FoundryDapDebugger *self = (FoundryDapDebugger *)object;
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  G_OBJECT_CLASS (foundry_dap_debugger_parent_class)->constructed (object);

  if (priv->subprocess != NULL)
    dex_future_disown (dex_future_finally (dex_subprocess_wait_check (priv->subprocess),
                                           foundry_dap_debugger_exited,
                                           foundry_weak_ref_new (self),
                                           (GDestroyNotify) foundry_weak_ref_free));

  if (priv->stream == NULL)
    {
      g_warning ("`%s` at %p created without a stream, this cannot work!",
                 G_OBJECT_TYPE_NAME (self), self);
      return;
    }

  priv->driver = foundry_dap_driver_new (priv->stream, FOUNDRY_JSONRPC_STYLE_HTTP);
  g_signal_connect_object (priv->driver,
                           "event",
                           G_CALLBACK (foundry_dap_debugger_driver_event_cb),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (priv->driver,
                           "handle-request",
                           G_CALLBACK (foundry_dap_debugger_driver_handle_request_cb),
                           self,
                           G_CONNECT_SWAPPED);
  foundry_dap_driver_start (priv->driver);
}

static void
foundry_dap_debugger_dispose (GObject *object)
{
  FoundryDapDebugger *self = (FoundryDapDebugger *)object;
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  if (priv->subprocess != NULL)
    g_subprocess_force_exit (priv->subprocess);

  if (priv->stream != NULL)
    g_io_stream_close (priv->stream, NULL, NULL);

  g_clear_pointer (&priv->trap_params, g_ptr_array_unref);
  g_clear_handle_id (&priv->sync_params_source, g_source_remove);

  g_clear_object (&priv->driver);
  g_clear_object (&priv->stream);
  g_clear_object (&priv->subprocess);
  g_clear_object (&priv->log_messages);
  g_clear_object (&priv->modules);
  g_clear_object (&priv->threads);
  g_clear_object (&priv->traps);
  g_clear_object (&priv->primary_thread);

  G_OBJECT_CLASS (foundry_dap_debugger_parent_class)->dispose (object);
}

static void
foundry_dap_debugger_get_property (GObject    *object,
                                   guint       prop_id,
                                   GValue     *value,
                                   GParamSpec *pspec)
{
  FoundryDapDebugger *self = FOUNDRY_DAP_DEBUGGER (object);

  switch (prop_id)
    {
    case PROP_QUIRKS:
      g_value_set_flags (value, foundry_dap_debugger_get_quirks (self));
      break;

    case PROP_STREAM:
      g_value_take_object (value, foundry_dap_debugger_dup_stream (self));
      break;

    case PROP_SUBPROCESS:
      g_value_take_object (value, foundry_dap_debugger_dup_subprocess (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_dap_debugger_set_property (GObject      *object,
                                   guint         prop_id,
                                   const GValue *value,
                                   GParamSpec   *pspec)
{
  FoundryDapDebugger *self = FOUNDRY_DAP_DEBUGGER (object);
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_QUIRKS:
      priv->quirks = g_value_get_flags (value);
      break;

    case PROP_STREAM:
      priv->stream = g_value_dup_object (value);
      break;

    case PROP_SUBPROCESS:
      priv->subprocess = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_dap_debugger_class_init (FoundryDapDebuggerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDebuggerClass *debugger_class = FOUNDRY_DEBUGGER_CLASS (klass);

  object_class->constructed = foundry_dap_debugger_constructed;
  object_class->dispose = foundry_dap_debugger_dispose;
  object_class->get_property = foundry_dap_debugger_get_property;
  object_class->set_property = foundry_dap_debugger_set_property;

  debugger_class->list_log_messages = foundry_dap_debugger_list_log_messages;
  debugger_class->list_modules = foundry_dap_debugger_list_modules;
  debugger_class->list_threads = foundry_dap_debugger_list_threads;
  debugger_class->list_traps = foundry_dap_debugger_list_traps;
  debugger_class->move = foundry_dap_debugger_move;
  debugger_class->interpret = foundry_dap_debugger_interpret;
  debugger_class->interrupt = foundry_dap_debugger_interrupt;
  debugger_class->trap = foundry_dap_debugger_trap;
  debugger_class->can_move = foundry_dap_debugger_can_move;
  debugger_class->disassemble = foundry_dap_debugger_disassemble;
  debugger_class->dup_primary_thread = foundry_dap_debugger_dup_primary_thread;
  debugger_class->stop = foundry_dap_debugger_stop;
  debugger_class->has_terminated = foundry_dap_debugger_has_terminated;

  properties[PROP_QUIRKS] =
    g_param_spec_flags ("quirks", NULL, NULL,
                        FOUNDRY_TYPE_DAP_DEBUGGER_QUIRK,
                        FOUNDRY_DAP_DEBUGGER_QUIRK_NONE,
                        (G_PARAM_READWRITE |
                         G_PARAM_CONSTRUCT_ONLY |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_STREAM] =
    g_param_spec_object ("stream", NULL, NULL,
                         G_TYPE_IO_STREAM,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SUBPROCESS] =
    g_param_spec_object ("subprocess", NULL, NULL,
                         G_TYPE_SUBPROCESS,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_dap_debugger_init (FoundryDapDebugger *self)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  priv->log_messages = g_list_store_new (FOUNDRY_TYPE_DAP_DEBUGGER_LOG_MESSAGE);
  priv->modules = g_list_store_new (FOUNDRY_TYPE_DAP_DEBUGGER_MODULE);
  priv->threads = g_list_store_new (FOUNDRY_TYPE_DAP_DEBUGGER_THREAD);
  priv->traps = g_list_store_new (FOUNDRY_TYPE_DEBUGGER_TRAP);
  priv->trap_params = g_ptr_array_new_with_free_func (g_object_unref);
}

/**
 * foundry_dap_debugger_dup_subprocess:
 * @self: a [class@Foundry.DapDebugger]
 *
 * Returns: (transfer full) (nullable):
 *
 * Since: 1.1
 */
GSubprocess *
foundry_dap_debugger_dup_subprocess (FoundryDapDebugger *self)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER (self), NULL);

  if (priv->subprocess)
    return g_object_ref (priv->subprocess);

  return NULL;
}

/**
 * foundry_dap_debugger_dup_stream:
 * @self: a [class@Foundry.DapDebugger]
 *
 * Returns: (transfer full) (nullable):
 *
 * Since: 1.1
 */
GIOStream *
foundry_dap_debugger_dup_stream (FoundryDapDebugger *self)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER (self), NULL);

  if (priv->stream)
    return g_object_ref (priv->stream);

  return NULL;
}

/**
 * foundry_dap_debugger_call:
 * @self: a [class@Foundry.DapDebugger]
 * @node: (transfer full):
 *
 * Makes a request to the DAP server. The reply will be provided
 * via the resulting future, even if the reply contains an error.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [struct@Json.Node] or rejects with error.
 *
 * Since: 1.1
 */
DexFuture *
foundry_dap_debugger_call (FoundryDapDebugger *self,
                           JsonNode           *node)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  dex_return_error_if_fail (FOUNDRY_IS_DAP_DEBUGGER (self));
  dex_return_error_if_fail (node != NULL);
  dex_return_error_if_fail (JSON_NODE_HOLDS_OBJECT (node));

  return foundry_dap_driver_call (priv->driver, node);
}

/**
 * foundry_dap_debugger_send:
 * @self: a [class@Foundry.DapDebugger]
 * @node: (transfer full):
 *
 * Send a message to the peer without expecting a reply.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error.
 *
 * Since: 1.1
 */
DexFuture *
foundry_dap_debugger_send (FoundryDapDebugger *self,
                           JsonNode           *node)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  dex_return_error_if_fail (FOUNDRY_IS_DAP_DEBUGGER (self));
  dex_return_error_if_fail (node != NULL);
  dex_return_error_if_fail (JSON_NODE_HOLDS_OBJECT (node));

  return foundry_dap_driver_send (priv->driver, node);
}

FoundryDapDebuggerQuirk
foundry_dap_debugger_get_quirks (FoundryDapDebugger *self)
{
  FoundryDapDebuggerPrivate *priv = foundry_dap_debugger_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER (self), 0);

  return priv->quirks;
}

DexFuture *
_foundry_dap_debugger_remove_breakpoint (FoundryDapDebugger *self,
                                         gint64              breakpoint_id)
{
  dex_return_error_if_fail (FOUNDRY_IS_DAP_DEBUGGER (self));
  dex_return_error_if_fail (breakpoint_id > 0);

  /* TODO: remove our params for the matching breakpoint and then sync
   *       breakpoints to the DAP server.
   */

  return dex_future_new_true ();
}

G_DEFINE_FLAGS_TYPE (FoundryDapDebuggerQuirk, foundry_dap_debugger_quirk,
                     G_DEFINE_ENUM_VALUE (FOUNDRY_DAP_DEBUGGER_QUIRK_NONE, "none"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_DAP_DEBUGGER_QUIRK_QUERY_THREADS, "query-threads"))
