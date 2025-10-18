/* foundry-dap-debugger-breakpoint.c
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

#include "foundry-dap-debugger-breakpoint-private.h"
#include "foundry-dap-debugger-private.h"
#include "foundry-json-node.h"

struct _FoundryDapDebuggerBreakpoint
{
  FoundryDebuggerBreakpoint parent_instance;
  GWeakRef debugger_wr;
  JsonNode *breakpoint_node;
  char *message;
  char *source_path;
  guint line;
  guint column;
  guint verified : 1;
};

enum {
  PROP_0,
  PROP_VERIFIED,
  PROP_MESSAGE,
  PROP_LINE,
  PROP_COLUMN,
  PROP_SOURCE_PATH,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryDapDebuggerBreakpoint, foundry_dap_debugger_breakpoint, FOUNDRY_TYPE_DEBUGGER_BREAKPOINT)

static GParamSpec *properties[N_PROPS];

static char *
foundry_dap_debugger_breakpoint_dup_id (FoundryDebuggerTrap *trap)
{
  FoundryDapDebuggerBreakpoint *self = FOUNDRY_DAP_DEBUGGER_BREAKPOINT (trap);
  gint64 id = 0;

  if (FOUNDRY_JSON_OBJECT_PARSE (self->breakpoint_node, "id", FOUNDRY_JSON_NODE_GET_INT (&id)))
    return g_strdup_printf ("%"G_GINT64_FORMAT, id);

  return g_strdup ("unknown");
}

static gboolean
foundry_dap_debugger_breakpoint_is_armed (FoundryDebuggerTrap *trap)
{
  FoundryDapDebuggerBreakpoint *self = FOUNDRY_DAP_DEBUGGER_BREAKPOINT (trap);

  return self->verified;
}

static DexFuture *
foundry_dap_debugger_breakpoint_arm (FoundryDebuggerTrap *trap)
{
  /* DAP breakpoints are managed by the DAP server, so we can't arm them directly */
  return dex_future_new_true ();
}

static DexFuture *
foundry_dap_debugger_breakpoint_disarm (FoundryDebuggerTrap *trap)
{
  /* DAP breakpoints are managed by the DAP server, so we can't disarm them directly */
  return dex_future_new_true ();
}

static DexFuture *
foundry_dap_debugger_breakpoint_remove (FoundryDebuggerTrap *trap)
{
  FoundryDapDebuggerBreakpoint *self = FOUNDRY_DAP_DEBUGGER_BREAKPOINT (trap);
  g_autoptr(FoundryDapDebugger) debugger = g_weak_ref_get (&self->debugger_wr);
  gint64 id = 0;

  if (debugger == NULL)
    return foundry_future_new_disposed ();

  if (FOUNDRY_JSON_OBJECT_PARSE (self->breakpoint_node, "id", FOUNDRY_JSON_NODE_GET_INT (&id)))
    return _foundry_dap_debugger_remove_breakpoint (debugger, id);

  return dex_future_new_true ();
}

static void
foundry_dap_debugger_breakpoint_get_property (GObject    *object,
                                              guint       prop_id,
                                              GValue     *value,
                                              GParamSpec *pspec)
{
  FoundryDapDebuggerBreakpoint *self = FOUNDRY_DAP_DEBUGGER_BREAKPOINT (object);

  switch (prop_id)
    {
    case PROP_VERIFIED:
      g_value_set_boolean (value, self->verified);
      break;

    case PROP_MESSAGE:
      g_value_set_string (value, self->message);
      break;

    case PROP_LINE:
      g_value_set_uint (value, self->line);
      break;

    case PROP_COLUMN:
      g_value_set_uint (value, self->column);
      break;

    case PROP_SOURCE_PATH:
      g_value_set_string (value, self->source_path);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_dap_debugger_breakpoint_finalize (GObject *object)
{
  FoundryDapDebuggerBreakpoint *self = FOUNDRY_DAP_DEBUGGER_BREAKPOINT (object);

  g_clear_pointer (&self->breakpoint_node, json_node_unref);
  g_clear_pointer (&self->message, g_free);
  g_clear_pointer (&self->source_path, g_free);

  g_weak_ref_clear (&self->debugger_wr);

  G_OBJECT_CLASS (foundry_dap_debugger_breakpoint_parent_class)->finalize (object);
}

static void
foundry_dap_debugger_breakpoint_class_init (FoundryDapDebuggerBreakpointClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDebuggerTrapClass *trap_class = FOUNDRY_DEBUGGER_TRAP_CLASS (klass);

  object_class->get_property = foundry_dap_debugger_breakpoint_get_property;
  object_class->finalize = foundry_dap_debugger_breakpoint_finalize;

  trap_class->dup_id = foundry_dap_debugger_breakpoint_dup_id;
  trap_class->is_armed = foundry_dap_debugger_breakpoint_is_armed;
  trap_class->arm = foundry_dap_debugger_breakpoint_arm;
  trap_class->disarm = foundry_dap_debugger_breakpoint_disarm;
  trap_class->remove = foundry_dap_debugger_breakpoint_remove;

  properties[PROP_VERIFIED] =
    g_param_spec_boolean ("verified", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_MESSAGE] =
    g_param_spec_string ("message", NULL, NULL,
                          NULL,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_LINE] =
    g_param_spec_uint ("line", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_COLUMN] =
    g_param_spec_uint ("column", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_SOURCE_PATH] =
    g_param_spec_string ("source-path", NULL, NULL,
                          NULL,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_dap_debugger_breakpoint_init (FoundryDapDebuggerBreakpoint *self)
{
  g_weak_ref_init (&self->debugger_wr, NULL);
}

FoundryDapDebuggerBreakpoint *
foundry_dap_debugger_breakpoint_new (FoundryDapDebugger *debugger,
                                     JsonNode           *breakpoint_node)
{
  FoundryDapDebuggerBreakpoint *self;
  const char *source_path = NULL;
  const char *message = NULL;
  JsonNode *source = NULL;
  gboolean verified = FALSE;
  gint64 line = 0;
  gint64 column = 0;

  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER (debugger), NULL);
  g_return_val_if_fail (breakpoint_node != NULL, NULL);
  g_return_val_if_fail (JSON_NODE_HOLDS_OBJECT (breakpoint_node), NULL);

  self = g_object_new (FOUNDRY_TYPE_DAP_DEBUGGER_BREAKPOINT, NULL);

  self->breakpoint_node = json_node_ref (breakpoint_node);
  g_weak_ref_set (&self->debugger_wr, debugger);

  /* Parse breakpoint properties */
  FOUNDRY_JSON_OBJECT_PARSE (breakpoint_node,
                             "verified", FOUNDRY_JSON_NODE_GET_BOOLEAN (&verified),
                             "message", FOUNDRY_JSON_NODE_GET_STRING (&message),
                             "line", FOUNDRY_JSON_NODE_GET_INT (&line),
                             "column", FOUNDRY_JSON_NODE_GET_INT (&column),
                             "source", FOUNDRY_JSON_NODE_GET_NODE (&source));

  self->verified = verified;
  self->message = g_strdup (message);
  self->line = (guint) line;
  self->column = (guint) column;

  if (source && FOUNDRY_JSON_OBJECT_PARSE (source, "path", FOUNDRY_JSON_NODE_GET_STRING (&source_path)))
    self->source_path = g_strdup (source_path);

  return self;
}

gboolean
foundry_dap_debugger_breakpoint_get_verified (FoundryDapDebuggerBreakpoint *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER_BREAKPOINT (self), FALSE);

  return self->verified;
}

char *
foundry_dap_debugger_breakpoint_dup_message (FoundryDapDebuggerBreakpoint *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER_BREAKPOINT (self), NULL);

  return g_strdup (self->message);
}

guint
foundry_dap_debugger_breakpoint_get_line (FoundryDapDebuggerBreakpoint *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER_BREAKPOINT (self), 0);

  return self->line;
}

guint
foundry_dap_debugger_breakpoint_get_column (FoundryDapDebuggerBreakpoint *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER_BREAKPOINT (self), 0);

  return self->column;
}

char *
foundry_dap_debugger_breakpoint_dup_source_path (FoundryDapDebuggerBreakpoint *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER_BREAKPOINT (self), NULL);

  return g_strdup (self->source_path);
}
