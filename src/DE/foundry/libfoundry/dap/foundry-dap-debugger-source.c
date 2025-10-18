/* foundry-dap-debugger-source.c
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

#include "foundry-dap-debugger-source-private.h"
#include "foundry-json-node.h"

struct _FoundryDapDebuggerSource
{
  FoundryDebuggerSource parent_instance;
  GWeakRef debugger_wr;
  JsonNode *node;
};

G_DEFINE_FINAL_TYPE (FoundryDapDebuggerSource, foundry_dap_debugger_source, FOUNDRY_TYPE_DEBUGGER_SOURCE)

static char *
foundry_dap_debugger_source_dup_id (FoundryDebuggerSource *source)
{
  FoundryDapDebuggerSource *self = FOUNDRY_DAP_DEBUGGER_SOURCE (source);
  const char *name;

  /* TODO: Check sourceReference <int> */

  if (FOUNDRY_JSON_OBJECT_PARSE (self->node, "name", FOUNDRY_JSON_NODE_GET_STRING (&name)))
    return g_strdup (name);

  return NULL;
}

static char *
foundry_dap_debugger_source_dup_name (FoundryDebuggerSource *source)
{
  FoundryDapDebuggerSource *self = FOUNDRY_DAP_DEBUGGER_SOURCE (source);
  const char *name;

  if (FOUNDRY_JSON_OBJECT_PARSE (self->node, "name", FOUNDRY_JSON_NODE_GET_STRING (&name)))
    return g_strdup (name);

  return NULL;
}

static char *
foundry_dap_debugger_source_dup_path (FoundryDebuggerSource *source)
{
  FoundryDapDebuggerSource *self = FOUNDRY_DAP_DEBUGGER_SOURCE (source);
  const char *path;

  if (FOUNDRY_JSON_OBJECT_PARSE (self->node, "path", FOUNDRY_JSON_NODE_GET_STRING (&path)))
    return g_strdup (path);

  return NULL;
}

static void
foundry_dap_debugger_source_finalize (GObject *object)
{
  FoundryDapDebuggerSource *self = (FoundryDapDebuggerSource *)object;

  g_weak_ref_clear (&self->debugger_wr);
  g_clear_pointer (&self->node, json_node_unref);

  G_OBJECT_CLASS (foundry_dap_debugger_source_parent_class)->finalize (object);
}

static void
foundry_dap_debugger_source_class_init (FoundryDapDebuggerSourceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDebuggerSourceClass *source_class = FOUNDRY_DEBUGGER_SOURCE_CLASS (klass);

  object_class->finalize = foundry_dap_debugger_source_finalize;

  source_class->dup_id = foundry_dap_debugger_source_dup_id;
  source_class->dup_name = foundry_dap_debugger_source_dup_name;
  source_class->dup_path = foundry_dap_debugger_source_dup_path;
}

static void
foundry_dap_debugger_source_init (FoundryDapDebuggerSource *self)
{
  g_weak_ref_init (&self->debugger_wr, NULL);
}

FoundryDebuggerSource *
foundry_dap_debugger_source_new (FoundryDapDebugger *debugger,
                                 JsonNode           *node)
{
  FoundryDapDebuggerSource *self = g_object_new (FOUNDRY_TYPE_DAP_DEBUGGER_SOURCE, NULL);

  g_weak_ref_set (&self->debugger_wr, debugger);
  self->node = json_node_ref (node);

  return FOUNDRY_DEBUGGER_SOURCE (self);
}
