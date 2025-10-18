/* foundry-debugger-stack-frame.c
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

#include "foundry-debugger-stack-frame.h"
#include "foundry-debugger-source.h"
#include "foundry-util.h"

enum {
  PROP_0,
  PROP_ID,
  PROP_NAME,
  PROP_MODULE_ID,
  PROP_INSTRUCTION_POINTER,
  PROP_CAN_RESTART,
  PROP_BEGIN_LINE,
  PROP_BEGIN_LINE_OFFSET,
  PROP_END_LINE,
  PROP_END_LINE_OFFSET,
  PROP_SOURCE,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryDebuggerStackFrame, foundry_debugger_stack_frame, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static guint
foundry_debugger_stack_frame_get_begin_line (FoundryDebuggerStackFrame *self)
{
  guint value;
  foundry_debugger_stack_frame_get_source_range (self, &value, NULL, NULL, NULL);
  return value;
}

static guint
foundry_debugger_stack_frame_get_begin_line_offset (FoundryDebuggerStackFrame *self)
{
  guint value;
  foundry_debugger_stack_frame_get_source_range (self, NULL, &value, NULL, NULL);
  return value;
}

static guint
foundry_debugger_stack_frame_get_end_line (FoundryDebuggerStackFrame *self)
{
  guint value;
  foundry_debugger_stack_frame_get_source_range (self, NULL, NULL, &value, NULL);
  return value;
}

static guint
foundry_debugger_stack_frame_get_end_line_offset (FoundryDebuggerStackFrame *self)
{
  guint value;
  foundry_debugger_stack_frame_get_source_range (self, NULL, NULL, NULL, &value);
  return value;
}

static void
foundry_debugger_stack_frame_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  FoundryDebuggerStackFrame *self = FOUNDRY_DEBUGGER_STACK_FRAME (object);

  switch (prop_id)
    {
    case PROP_CAN_RESTART:
      g_value_set_boolean (value, foundry_debugger_stack_frame_can_restart (self));
      break;

    case PROP_ID:
      g_value_take_string (value, foundry_debugger_stack_frame_dup_id (self));
      break;

    case PROP_NAME:
      g_value_take_string (value, foundry_debugger_stack_frame_dup_name (self));
      break;

    case PROP_MODULE_ID:
      g_value_take_string (value, foundry_debugger_stack_frame_dup_module_id (self));
      break;

    case PROP_INSTRUCTION_POINTER:
      g_value_set_uint64 (value, foundry_debugger_stack_frame_get_instruction_pointer (self));
      break;

    case PROP_BEGIN_LINE:
      g_value_set_uint (value, foundry_debugger_stack_frame_get_begin_line (self));
      break;

    case PROP_BEGIN_LINE_OFFSET:
      g_value_set_uint (value, foundry_debugger_stack_frame_get_begin_line_offset (self));
      break;

    case PROP_END_LINE:
      g_value_set_uint (value, foundry_debugger_stack_frame_get_end_line (self));
      break;

    case PROP_END_LINE_OFFSET:
      g_value_set_uint (value, foundry_debugger_stack_frame_get_end_line_offset (self));
      break;

    case PROP_SOURCE:
      g_value_take_object (value, foundry_debugger_stack_frame_dup_source (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_stack_frame_class_init (FoundryDebuggerStackFrameClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_stack_frame_get_property;

  properties[PROP_CAN_RESTART] =
    g_param_spec_boolean ("can-restart", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_MODULE_ID] =
    g_param_spec_string ("module-id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_INSTRUCTION_POINTER] =
    g_param_spec_uint64 ("instruction-pointer", NULL, NULL,
                         0, G_MAXUINT64, 0,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_BEGIN_LINE] =
    g_param_spec_uint ("begin-line", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_BEGIN_LINE_OFFSET] =
    g_param_spec_uint ("begin-line-offset", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_END_LINE] =
    g_param_spec_uint ("end-line", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_END_LINE_OFFSET] =
    g_param_spec_uint ("end-line-offset", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  /**
   * FoundryDebuggerStackFrame:source:
   *
   * Since: 1.1
   */
  properties[PROP_SOURCE] =
    g_param_spec_object ("source", NULL, NULL,
                         FOUNDRY_TYPE_DEBUGGER_SOURCE,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_stack_frame_init (FoundryDebuggerStackFrame *self)
{
}

char *
foundry_debugger_stack_frame_dup_id (FoundryDebuggerStackFrame *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_STACK_FRAME (self), NULL);

  if (FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->dup_id)
    return FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->dup_id (self);

  return NULL;
}

char *
foundry_debugger_stack_frame_dup_name (FoundryDebuggerStackFrame *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_STACK_FRAME (self), NULL);

  if (FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->dup_name)
    return FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->dup_name (self);

  return NULL;
}

char *
foundry_debugger_stack_frame_dup_module_id (FoundryDebuggerStackFrame *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_STACK_FRAME (self), NULL);

  if (FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->dup_module_id)
    return FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->dup_module_id (self);

  return NULL;
}

guint64
foundry_debugger_stack_frame_get_instruction_pointer (FoundryDebuggerStackFrame *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_STACK_FRAME (self), 0);

  if (FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->get_instruction_pointer)
    return FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->get_instruction_pointer (self);

  return 0;
}

gboolean
foundry_debugger_stack_frame_can_restart (FoundryDebuggerStackFrame *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_STACK_FRAME (self), FALSE);

  if (FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->can_restart)
    return FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->can_restart (self);

  return FALSE;
}

void
foundry_debugger_stack_frame_get_source_range (FoundryDebuggerStackFrame *self,
                                               guint                     *begin_line,
                                               guint                     *begin_line_offset,
                                               guint                     *end_line,
                                               guint                     *end_line_offset)
{
  guint dummy1;
  guint dummy2;
  guint dummy3;
  guint dummy4;

  g_return_if_fail (FOUNDRY_IS_DEBUGGER_STACK_FRAME (self));

  if (begin_line == NULL)
    begin_line = &dummy1;

  if (begin_line_offset == NULL)
    begin_line_offset = &dummy2;

  if (end_line == NULL)
    end_line = &dummy3;

  if (end_line_offset == NULL)
    end_line_offset = &dummy4;

  if (FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->get_source_range)
    FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->get_source_range (self,
                                                                     begin_line,
                                                                     begin_line_offset,
                                                                     end_line,
                                                                     end_line_offset);
}

/**
 * foundry_debugger_stack_frame_dup_source:
 * @self: a [class@Foundry.DebuggerStackFrame]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryDebuggerSource *
foundry_debugger_stack_frame_dup_source (FoundryDebuggerStackFrame *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_STACK_FRAME (self), NULL);

  if (FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->dup_source)
    return FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->dup_source (self);

  return NULL;
}

/**
 * foundry_debugger_stack_frame_list_params:
 * @self: a [class@Foundry.DebuggerStackFrame]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.DebuggerVariable]
 */
DexFuture *
foundry_debugger_stack_frame_list_params (FoundryDebuggerStackFrame *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER_STACK_FRAME (self));

  if (FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->list_params)
    return FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->list_params (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_stack_frame_list_locals:
 * @self: a [class@Foundry.DebuggerStackFrame]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.DebuggerVariable]
 */
DexFuture *
foundry_debugger_stack_frame_list_locals (FoundryDebuggerStackFrame *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER_STACK_FRAME (self));

  if (FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->list_locals)
    return FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->list_locals (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_stack_frame_list_registers:
 * @self: a [class@Foundry.DebuggerStackFrame]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.DebuggerVariable]
 */
DexFuture *
foundry_debugger_stack_frame_list_registers (FoundryDebuggerStackFrame *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER_STACK_FRAME (self));

  if (FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->list_registers)
    return FOUNDRY_DEBUGGER_STACK_FRAME_GET_CLASS (self)->list_registers (self);

  return foundry_future_new_not_supported ();
}
