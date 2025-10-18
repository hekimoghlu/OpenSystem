/* foundry-debugger-log-message.c
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

#include "foundry-debugger-log-message.h"

enum {
  PROP_0,
  PROP_MESSAGE,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryDebuggerLogMessage, foundry_debugger_log_message, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_log_message_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  FoundryDebuggerLogMessage *self = FOUNDRY_DEBUGGER_LOG_MESSAGE (object);

  switch (prop_id)
    {
    case PROP_MESSAGE:
      g_value_take_string (value, foundry_debugger_log_message_dup_message (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_log_message_class_init (FoundryDebuggerLogMessageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_log_message_get_property;

  properties[PROP_MESSAGE] =
    g_param_spec_string ("message", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_log_message_init (FoundryDebuggerLogMessage *self)
{
}

/**
 * foundry_debugger_log_message_dup_message:
 * @self: a [class@Foundry.DebuggerLogMessage]
 *
 * The message string from the debugger instance.
 *
 * Returns: (transfer full) (nullable):
 *
 * Since: 1.1
 */
char *
foundry_debugger_log_message_dup_message (FoundryDebuggerLogMessage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_LOG_MESSAGE (self), NULL);

  if (FOUNDRY_DEBUGGER_LOG_MESSAGE_GET_CLASS (self)->dup_message)
    return FOUNDRY_DEBUGGER_LOG_MESSAGE_GET_CLASS (self)->dup_message (self);

  return NULL;
}
