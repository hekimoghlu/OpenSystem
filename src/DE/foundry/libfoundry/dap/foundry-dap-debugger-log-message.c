/* foundry-dap-debugger-log-message.c
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

#include "foundry-dap-debugger-log-message-private.h"
#include "foundry-json-node.h"

struct _FoundryDapDebuggerLogMessage
{
  FoundryDebuggerLogMessage parent_instance;
  JsonNode *node;
};

G_DEFINE_FINAL_TYPE (FoundryDapDebuggerLogMessage, foundry_dap_debugger_log_message, FOUNDRY_TYPE_DEBUGGER_LOG_MESSAGE)

static char *
foundry_dap_debugger_log_message_dup_message (FoundryDebuggerLogMessage *message)
{
  FoundryDapDebuggerLogMessage *self = FOUNDRY_DAP_DEBUGGER_LOG_MESSAGE (message);
  const char *output = NULL;

  if (FOUNDRY_JSON_OBJECT_PARSE (self->node,
                                 "body", "{",
                                   "output", FOUNDRY_JSON_NODE_GET_STRING (&output),
                                 "}"))
    return g_strchomp (g_strdup (output));

  return NULL;
}

static void
foundry_dap_debugger_log_message_finalize (GObject *object)
{
  FoundryDapDebuggerLogMessage *self = (FoundryDapDebuggerLogMessage *)object;

  g_clear_pointer (&self->node, json_node_unref);

  G_OBJECT_CLASS (foundry_dap_debugger_log_message_parent_class)->finalize (object);
}

static void
foundry_dap_debugger_log_message_class_init (FoundryDapDebuggerLogMessageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDebuggerLogMessageClass *log_message_class = FOUNDRY_DEBUGGER_LOG_MESSAGE_CLASS (klass);

  object_class->finalize = foundry_dap_debugger_log_message_finalize;

  log_message_class->dup_message = foundry_dap_debugger_log_message_dup_message;
}

static void
foundry_dap_debugger_log_message_init (FoundryDapDebuggerLogMessage *self)
{
}

FoundryDebuggerLogMessage *
foundry_dap_debugger_log_message_new (JsonNode *node)
{
  FoundryDapDebuggerLogMessage *self;

  g_return_val_if_fail (node != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_DAP_DEBUGGER_LOG_MESSAGE, NULL);
  self->node = json_node_ref (node);

  return FOUNDRY_DEBUGGER_LOG_MESSAGE (self);
}
