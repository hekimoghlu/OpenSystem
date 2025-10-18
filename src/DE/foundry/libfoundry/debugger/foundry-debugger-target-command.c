/* foundry-debugger-target-command.c
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
#include "foundry-debugger-target-command.h"
#include "foundry-debugger-target-private.h"

struct _FoundryDebuggerTargetCommand
{
  FoundryDebuggerTarget parent_instance;
  FoundryCommand *command;
};

struct _FoundryDebuggerTargetCommandClass
{
  FoundryDebuggerTargetClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryDebuggerTargetCommand, foundry_debugger_target_command, FOUNDRY_TYPE_DEBUGGER_TARGET)

enum {
  PROP_0,
  PROP_COMMAND,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_target_command_finalize (GObject *object)
{
  FoundryDebuggerTargetCommand *self = (FoundryDebuggerTargetCommand *)object;

  g_clear_object (&self->command);

  G_OBJECT_CLASS (foundry_debugger_target_command_parent_class)->finalize (object);
}

static void
foundry_debugger_target_command_get_property (GObject    *object,
                                              guint       prop_id,
                                              GValue     *value,
                                              GParamSpec *pspec)
{
  FoundryDebuggerTargetCommand *self = FOUNDRY_DEBUGGER_TARGET_COMMAND (object);

  switch (prop_id)
    {
    case PROP_COMMAND:
      g_value_take_object (value, foundry_debugger_target_command_dup_command (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_target_command_set_property (GObject      *object,
                                              guint         prop_id,
                                              const GValue *value,
                                              GParamSpec   *pspec)
{
  FoundryDebuggerTargetCommand *self = FOUNDRY_DEBUGGER_TARGET_COMMAND (object);

  switch (prop_id)
    {
    case PROP_COMMAND:
      self->command = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_target_command_class_init (FoundryDebuggerTargetCommandClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_debugger_target_command_finalize;
  object_class->get_property = foundry_debugger_target_command_get_property;
  object_class->set_property = foundry_debugger_target_command_set_property;

  properties[PROP_COMMAND] =
    g_param_spec_object ("command", NULL, NULL,
                         FOUNDRY_TYPE_COMMAND,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_target_command_init (FoundryDebuggerTargetCommand *self)
{
}

/**
 * foundry_debugger_target_command_new:
 * @command: a [class@Foundry.Command]
 *
 * Returns: (transfer full):
 *
 * Since: 1.1
 */
FoundryDebuggerTarget *
foundry_debugger_target_command_new (FoundryCommand *command)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND (command), NULL);

  return g_object_new (FOUNDRY_TYPE_DEBUGGER_TARGET_COMMAND,
                       "command", command,
                       NULL);
}

/**
 * foundry_debugger_target_command_dup_command:
 * @self: a [class@Foundry.DebuggerTargetCommand]
 *
 * Returns: (transfer full):
 */
FoundryCommand *
foundry_debugger_target_command_dup_command (FoundryDebuggerTargetCommand *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TARGET_COMMAND (self), NULL);

  return g_object_ref (self->command);
}
