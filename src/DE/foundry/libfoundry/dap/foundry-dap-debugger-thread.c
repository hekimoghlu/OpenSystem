/* foundry-dap-debugger-thread.c
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

#include "foundry-dap-debugger-private.h"
#include "foundry-dap-debugger-thread-private.h"
#include "foundry-dap-debugger-stack-frame-private.h"
#include "foundry-dap-protocol.h"
#include "foundry-json-node.h"
#include "foundry-util.h"

struct _FoundryDapDebuggerThread
{
  FoundryDebuggerThread parent_instance;
  GWeakRef debugger_wr;
  gint64 id;
  guint stopped : 1;
};

G_DEFINE_FINAL_TYPE (FoundryDapDebuggerThread, foundry_dap_debugger_thread, FOUNDRY_TYPE_DEBUGGER_THREAD)

static char *
foundry_dap_debugger_thread_dup_id (FoundryDebuggerThread *thread)
{
  return g_strdup_printf ("%"G_GINT64_FORMAT, FOUNDRY_DAP_DEBUGGER_THREAD (thread)->id);
}

static DexFuture *
foundry_dap_debugger_thread_inflate_frames (DexFuture *future,
                                            gpointer   user_data)
{
  FoundryDapDebuggerThread *self = user_data;
  g_autoptr(FoundryDapDebugger) debugger = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(JsonNode) node = NULL;
  JsonNode *stack_frames = NULL;
  JsonArray *ar;
  guint length;

  g_assert (DEX_IS_FUTURE (future));
  g_assert (FOUNDRY_IS_DAP_DEBUGGER_THREAD (self));

  if (!(node = dex_await_boxed (dex_ref (future), NULL)))
    return dex_ref (future);

  if (!(debugger = g_weak_ref_get (&self->debugger_wr)))
    return foundry_future_new_disposed ();

  if (!FOUNDRY_JSON_OBJECT_PARSE (node,
                                  "body", "{",
                                    "stackFrames", FOUNDRY_JSON_NODE_GET_NODE (&stack_frames),
                                  "}") ||
      !JSON_NODE_HOLDS_ARRAY (stack_frames))
    return foundry_future_new_not_supported ();

  ar = json_node_get_array (stack_frames);
  length = json_array_get_length (ar);
  store = g_list_store_new (FOUNDRY_TYPE_DEBUGGER_STACK_FRAME);

  for (guint i = 0; i < length; i++)
    {
      JsonNode *stack_frame = json_array_get_element (ar, i);
      g_autoptr(FoundryDebuggerStackFrame) item = NULL;

      if (!JSON_NODE_HOLDS_OBJECT (stack_frame))
        continue;

      if ((item = foundry_dap_debugger_stack_frame_new (debugger, stack_frame)))
        g_list_store_append (store, item);
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static DexFuture *
foundry_dap_debugger_thread_list_frames (FoundryDebuggerThread *thread)
{
  FoundryDapDebuggerThread *self = FOUNDRY_DAP_DEBUGGER_THREAD (thread);
  g_autoptr(FoundryDapDebugger) debugger = g_weak_ref_get (&self->debugger_wr);

  if (debugger == NULL)
    return foundry_future_new_disposed ();

  return dex_future_then (foundry_dap_debugger_call (debugger,
                                                     FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                              "command", "stackTrace",
                                                                              "arguments", "{",
                                                                                "threadId", FOUNDRY_JSON_NODE_PUT_INT (self->id),
                                                                              "}")),
                          foundry_dap_debugger_thread_inflate_frames,
                          g_object_ref (self),
                          g_object_unref);
}

static gboolean
foundry_dap_debugger_thread_is_stopped (FoundryDebuggerThread *thread)
{
  return FOUNDRY_DAP_DEBUGGER_THREAD (thread)->stopped;
}

static gboolean
foundry_dap_debugger_thread_can_move (FoundryDebuggerThread   *thread,
                                      FoundryDebuggerMovement  movement)
{
  FoundryDapDebuggerThread *self = (FoundryDapDebuggerThread *)thread;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER_THREAD (self));

  if (movement == FOUNDRY_DEBUGGER_MOVEMENT_START)
    return FALSE;

  return self->stopped;
}

static DexFuture *
foundry_dap_debugger_thread_move (FoundryDebuggerThread   *thread,
                                  FoundryDebuggerMovement  movement)
{
  FoundryDapDebuggerThread *self = (FoundryDapDebuggerThread *)thread;
  g_autoptr(FoundryDapDebugger) debugger = NULL;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER_THREAD (self));


  if ((debugger = g_weak_ref_get (&self->debugger_wr)))
    return _foundry_dap_debugger_move (debugger, self->id, movement);

  return foundry_future_new_disposed ();
}

static DexFuture *
foundry_dap_debugger_thread_interrupt (FoundryDebuggerThread *thread)
{
  FoundryDapDebuggerThread *self = (FoundryDapDebuggerThread *)thread;
  g_autoptr(FoundryDapDebugger) debugger = NULL;

  g_assert (FOUNDRY_IS_DAP_DEBUGGER_THREAD (self));

  if ((debugger = g_weak_ref_get (&self->debugger_wr)))
    return dex_future_then (foundry_dap_debugger_call (debugger,
                                                       FOUNDRY_JSON_OBJECT_NEW ("type", "request",
                                                                                "command", "pause",
                                                                                "arguments", "{",
                                                                                "threadId", FOUNDRY_JSON_NODE_PUT_INT (self->id),
                                                                                "}")),
                            foundry_dap_protocol_unwrap_error,
                            NULL, NULL);

  return foundry_future_new_disposed ();
}

static void
foundry_dap_debugger_thread_finalize (GObject *object)
{
  FoundryDapDebuggerThread *self = (FoundryDapDebuggerThread *)object;

  g_weak_ref_clear (&self->debugger_wr);

  G_OBJECT_CLASS (foundry_dap_debugger_thread_parent_class)->finalize (object);
}

static void
foundry_dap_debugger_thread_class_init (FoundryDapDebuggerThreadClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDebuggerThreadClass *thread_class = FOUNDRY_DEBUGGER_THREAD_CLASS (klass);

  object_class->finalize = foundry_dap_debugger_thread_finalize;

  thread_class->dup_id = foundry_dap_debugger_thread_dup_id;
  thread_class->list_frames = foundry_dap_debugger_thread_list_frames;
  thread_class->is_stopped = foundry_dap_debugger_thread_is_stopped;
  thread_class->move = foundry_dap_debugger_thread_move;
  thread_class->can_move = foundry_dap_debugger_thread_can_move;
  thread_class->interrupt = foundry_dap_debugger_thread_interrupt;
}

static void
foundry_dap_debugger_thread_init (FoundryDapDebuggerThread *self)
{
  g_weak_ref_init (&self->debugger_wr, NULL);
}

FoundryDebuggerThread *
foundry_dap_debugger_thread_new (FoundryDapDebugger *debugger,
                                 gint64              id)
{
  FoundryDapDebuggerThread *self;

  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER (debugger), NULL);

  self = g_object_new (FOUNDRY_TYPE_DAP_DEBUGGER_THREAD, NULL);
  g_weak_ref_set (&self->debugger_wr, debugger);
  self->id = id;

  return FOUNDRY_DEBUGGER_THREAD (self);
}

void
foundry_dap_debugger_thread_set_stopped (FoundryDapDebuggerThread *self,
                                         gboolean                  stopped)
{
  g_return_if_fail (FOUNDRY_IS_DAP_DEBUGGER_THREAD (self));

  stopped = !!stopped;

  if (stopped != self->stopped)
    {
      self->stopped = stopped;
      g_object_notify (G_OBJECT (self), "stopped");
    }
}

gint64
foundry_dap_debugger_thread_get_id (FoundryDapDebuggerThread *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER_THREAD (self), 0);

  return self->id;
}
