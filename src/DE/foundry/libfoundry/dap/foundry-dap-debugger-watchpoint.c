/* foundry-dap-debugger-watchpoint.c
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

#include "foundry-dap-debugger-watchpoint-private.h"
#include "foundry-dap-debugger-private.h"
#include "foundry-json-node.h"

struct _FoundryDapDebuggerWatchpoint
{
  FoundryDebuggerWatchpoint parent_instance;
  GWeakRef debugger_wr;
  JsonNode *breakpoint_node;
  char *message;
  char *data_id;
  char *access_type;
  guint verified : 1;
};

enum {
  PROP_0,
  PROP_VERIFIED,
  PROP_MESSAGE,
  PROP_DATA_ID,
  PROP_ACCESS_TYPE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryDapDebuggerWatchpoint, foundry_dap_debugger_watchpoint, FOUNDRY_TYPE_DEBUGGER_WATCHPOINT)

static GParamSpec *properties[N_PROPS];

static char *
foundry_dap_debugger_watchpoint_dup_id (FoundryDebuggerTrap *trap)
{
  FoundryDapDebuggerWatchpoint *self = FOUNDRY_DAP_DEBUGGER_WATCHPOINT (trap);
  gint64 id = 0;

  if (FOUNDRY_JSON_OBJECT_PARSE (self->breakpoint_node, "id", FOUNDRY_JSON_NODE_GET_INT (&id)))
    return g_strdup_printf ("%"G_GINT64_FORMAT, id);

  return g_strdup ("unknown");
}

static gboolean
foundry_dap_debugger_watchpoint_is_armed (FoundryDebuggerTrap *trap)
{
  FoundryDapDebuggerWatchpoint *self = FOUNDRY_DAP_DEBUGGER_WATCHPOINT (trap);

  return self->verified;
}

static DexFuture *
foundry_dap_debugger_watchpoint_arm (FoundryDebuggerTrap *trap)
{
  /* TODO: */
  return dex_future_new_true ();
}

static DexFuture *
foundry_dap_debugger_watchpoint_disarm (FoundryDebuggerTrap *trap)
{
  /* TODO: */
  return dex_future_new_true ();
}

static DexFuture *
foundry_dap_debugger_watchpoint_remove (FoundryDebuggerTrap *trap)
{
  FoundryDapDebuggerWatchpoint *self = FOUNDRY_DAP_DEBUGGER_WATCHPOINT (trap);
  g_autoptr(FoundryDapDebugger) debugger = g_weak_ref_get (&self->debugger_wr);
  gint64 id = 0;

  if (debugger == NULL)
    return foundry_future_new_disposed ();

  if (FOUNDRY_JSON_OBJECT_PARSE (self->breakpoint_node, "id", FOUNDRY_JSON_NODE_GET_INT (&id)))
    return _foundry_dap_debugger_remove_breakpoint (debugger, id);

  return dex_future_new_true ();
}

static void
foundry_dap_debugger_watchpoint_get_property (GObject    *object,
                                              guint       prop_id,
                                              GValue     *value,
                                              GParamSpec *pspec)
{
  FoundryDapDebuggerWatchpoint *self = FOUNDRY_DAP_DEBUGGER_WATCHPOINT (object);

  switch (prop_id)
    {
    case PROP_VERIFIED:
      g_value_set_boolean (value, self->verified);
      break;

    case PROP_MESSAGE:
      g_value_set_string (value, self->message);
      break;

    case PROP_DATA_ID:
      g_value_set_string (value, self->data_id);
      break;

    case PROP_ACCESS_TYPE:
      g_value_set_string (value, self->access_type);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_dap_debugger_watchpoint_finalize (GObject *object)
{
  FoundryDapDebuggerWatchpoint *self = FOUNDRY_DAP_DEBUGGER_WATCHPOINT (object);

  g_clear_pointer (&self->breakpoint_node, json_node_unref);
  g_clear_pointer (&self->message, g_free);
  g_clear_pointer (&self->data_id, g_free);
  g_clear_pointer (&self->access_type, g_free);

  g_weak_ref_clear (&self->debugger_wr);

  G_OBJECT_CLASS (foundry_dap_debugger_watchpoint_parent_class)->finalize (object);
}

static void
foundry_dap_debugger_watchpoint_class_init (FoundryDapDebuggerWatchpointClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDebuggerTrapClass *trap_class = FOUNDRY_DEBUGGER_TRAP_CLASS (klass);

  object_class->get_property = foundry_dap_debugger_watchpoint_get_property;
  object_class->finalize = foundry_dap_debugger_watchpoint_finalize;

  trap_class->dup_id = foundry_dap_debugger_watchpoint_dup_id;
  trap_class->is_armed = foundry_dap_debugger_watchpoint_is_armed;
  trap_class->arm = foundry_dap_debugger_watchpoint_arm;
  trap_class->disarm = foundry_dap_debugger_watchpoint_disarm;
  trap_class->remove = foundry_dap_debugger_watchpoint_remove;

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

  properties[PROP_DATA_ID] =
    g_param_spec_string ("data-id", NULL, NULL,
                          NULL,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_ACCESS_TYPE] =
    g_param_spec_string ("access-type", NULL, NULL,
                          NULL,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_dap_debugger_watchpoint_init (FoundryDapDebuggerWatchpoint *self)
{
  g_weak_ref_init (&self->debugger_wr, NULL);
}

FoundryDapDebuggerWatchpoint *
foundry_dap_debugger_watchpoint_new (FoundryDapDebugger *debugger,
                                     JsonNode           *breakpoint_node)
{
  FoundryDapDebuggerWatchpoint *self;
  gboolean verified = FALSE;
  const char *message = NULL;

  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER (debugger), NULL);
  g_return_val_if_fail (breakpoint_node != NULL, NULL);
  g_return_val_if_fail (JSON_NODE_HOLDS_OBJECT (breakpoint_node), NULL);

  self = g_object_new (FOUNDRY_TYPE_DAP_DEBUGGER_WATCHPOINT, NULL);
  self->breakpoint_node = json_node_ref (breakpoint_node);
  g_weak_ref_set (&self->debugger_wr, debugger);

  FOUNDRY_JSON_OBJECT_PARSE (breakpoint_node,
                             "verified", FOUNDRY_JSON_NODE_GET_BOOLEAN (&verified),
                             "message", FOUNDRY_JSON_NODE_GET_STRING (&message));

  self->verified = verified;
  self->message = g_strdup (message);

  return self;
}

gboolean
foundry_dap_debugger_watchpoint_get_verified (FoundryDapDebuggerWatchpoint *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER_WATCHPOINT (self), FALSE);

  return self->verified;
}

char *
foundry_dap_debugger_watchpoint_dup_message (FoundryDapDebuggerWatchpoint *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER_WATCHPOINT (self), NULL);

  return g_strdup (self->message);
}

char *
foundry_dap_debugger_watchpoint_dup_data_id (FoundryDapDebuggerWatchpoint *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER_WATCHPOINT (self), NULL);

  return g_strdup (self->data_id);
}

char *
foundry_dap_debugger_watchpoint_dup_access_type (FoundryDapDebuggerWatchpoint *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER_WATCHPOINT (self), NULL);

  return g_strdup (self->access_type);
}
