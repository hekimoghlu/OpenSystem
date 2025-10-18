/* foundry-debugger-stop-event.c
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

#include "foundry-debugger-stop-event.h"
#include "foundry-debugger-trap.h"

enum {
  PROP_0,
  PROP_REASON,
  PROP_TRAP,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryDebuggerStopEvent, foundry_debugger_stop_event, FOUNDRY_TYPE_DEBUGGER_EVENT)

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_stop_event_get_property (GObject    *object,
                                          guint       prop_id,
                                          GValue     *value,
                                          GParamSpec *pspec)
{
  FoundryDebuggerStopEvent *self = FOUNDRY_DEBUGGER_STOP_EVENT (object);

  switch (prop_id)
    {
    case PROP_REASON:
      g_value_set_enum (value, foundry_debugger_stop_event_get_reason (self));
      break;

    case PROP_TRAP:
      g_value_take_object (value, foundry_debugger_stop_event_dup_trap (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_stop_event_class_init (FoundryDebuggerStopEventClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_stop_event_get_property;

  properties[PROP_REASON] =
    g_param_spec_enum ("reason", NULL, NULL,
                       FOUNDRY_TYPE_DEBUGGER_STOP_REASON,
                       FOUNDRY_DEBUGGER_STOP_UNKNOWN,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_TRAP] =
    g_param_spec_object ("trap", NULL, NULL,
                         FOUNDRY_TYPE_DEBUGGER_TRAP,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_stop_event_init (FoundryDebuggerStopEvent *self)
{
}

/**
 * foundry_debugger_stop_event_dup_trap:
 * @self: a [class@Foundry.DebuggerStopEvent]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryDebuggerTrap *
foundry_debugger_stop_event_dup_trap (FoundryDebuggerStopEvent *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_STOP_EVENT (self), NULL);

  if (FOUNDRY_DEBUGGER_STOP_EVENT_GET_CLASS (self)->dup_trap)
    return FOUNDRY_DEBUGGER_STOP_EVENT_GET_CLASS (self)->dup_trap (self);

  return NULL;
}

FoundryDebuggerStopReason
foundry_debugger_stop_event_get_reason (FoundryDebuggerStopEvent *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_STOP_EVENT (self), 0);

  if (FOUNDRY_DEBUGGER_STOP_EVENT_GET_CLASS (self)->get_reason)
    return FOUNDRY_DEBUGGER_STOP_EVENT_GET_CLASS (self)->get_reason (self);

  return FOUNDRY_DEBUGGER_STOP_UNKNOWN;
}

int
foundry_debugger_stop_event_get_signal (FoundryDebuggerStopEvent *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_STOP_EVENT (self), 0);

  if (FOUNDRY_DEBUGGER_STOP_EVENT_GET_CLASS (self)->get_signal)
    return FOUNDRY_DEBUGGER_STOP_EVENT_GET_CLASS (self)->get_signal (self);

  return 0;
}

int
foundry_debugger_stop_event_get_exit_code (FoundryDebuggerStopEvent *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_STOP_EVENT (self), 0);

  if (FOUNDRY_DEBUGGER_STOP_EVENT_GET_CLASS (self)->get_exit_code)
    return FOUNDRY_DEBUGGER_STOP_EVENT_GET_CLASS (self)->get_exit_code (self);

  return 0;
}

G_DEFINE_ENUM_TYPE (FoundryDebuggerStopReason, foundry_debugger_stop_reason,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_STOP_BREAKPOINT_HIT, "breakpoint-hit"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_STOP_EXITED, "exited"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_STOP_EXITED_NORMALLY, "exited-normally"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_STOP_EXITED_SIGNALED, "signaled"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_STOP_FUNCTION_FINISHED, "function-finished"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_STOP_LOCATION_REACHED, "location-reached"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_STOP_SIGNAL_RECEIVED, "signal-received"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_STOP_CATCH, "catch"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_STOP_UNKNOWN, "unknown"))
