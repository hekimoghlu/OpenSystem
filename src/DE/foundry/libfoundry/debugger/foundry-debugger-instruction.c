/* foundry-debugger-instruction.c
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

#include "foundry-debugger-instruction.h"

enum {
  PROP_0,
  PROP_DISPLAY_TEXT,
  PROP_FUNCTION,
  PROP_INSTRUCTION_POINTER,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryDebuggerInstruction, foundry_debugger_instruction, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_instruction_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  FoundryDebuggerInstruction *self = FOUNDRY_DEBUGGER_INSTRUCTION (object);

  switch (prop_id)
    {
    case PROP_DISPLAY_TEXT:
      g_value_take_string (value, foundry_debugger_instruction_dup_display_text (self));
      break;

    case PROP_FUNCTION:
      g_value_take_string (value, foundry_debugger_instruction_dup_function (self));
      break;

    case PROP_INSTRUCTION_POINTER:
      g_value_set_uint64 (value, foundry_debugger_instruction_get_instruction_pointer (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_instruction_class_init (FoundryDebuggerInstructionClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_instruction_get_property;

  properties[PROP_DISPLAY_TEXT] =
    g_param_spec_string ("display-text", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_FUNCTION] =
    g_param_spec_string ("function", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_INSTRUCTION_POINTER] =
    g_param_spec_uint64 ("instruction-pointer", NULL, NULL,
                         0, G_MAXUINT64, 0,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_instruction_init (FoundryDebuggerInstruction *self)
{
}

char *
foundry_debugger_instruction_dup_display_text (FoundryDebuggerInstruction *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_INSTRUCTION (self), NULL);

  if (FOUNDRY_DEBUGGER_INSTRUCTION_GET_CLASS (self)->dup_display_text)
    return FOUNDRY_DEBUGGER_INSTRUCTION_GET_CLASS (self)->dup_display_text (self);

  return NULL;
}

char *
foundry_debugger_instruction_dup_function (FoundryDebuggerInstruction *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_INSTRUCTION (self), NULL);

  if (FOUNDRY_DEBUGGER_INSTRUCTION_GET_CLASS (self)->dup_function)
    return FOUNDRY_DEBUGGER_INSTRUCTION_GET_CLASS (self)->dup_function (self);

  return NULL;
}

guint64
foundry_debugger_instruction_get_instruction_pointer (FoundryDebuggerInstruction *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_INSTRUCTION (self), 0);

  if (FOUNDRY_DEBUGGER_INSTRUCTION_GET_CLASS (self)->get_instruction_pointer)
    return FOUNDRY_DEBUGGER_INSTRUCTION_GET_CLASS (self)->get_instruction_pointer (self);

  return 0;
}
