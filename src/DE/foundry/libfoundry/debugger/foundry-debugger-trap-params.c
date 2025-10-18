/* foundry-debugger-trap-params.c
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

#include "foundry-debugger-trap-params.h"

struct _FoundryDebuggerTrapParams
{
  GObject                         parent_instance;
  char                           *path;
  char                           *function;
  char                           *thread_id;
  char                           *stack_frame_id;
  guint64                         instruction_pointer;
  guint                           line;
  guint                           line_offset;
  FoundryDebuggerTrapDisposition  disposition : 4;
  FoundryDebuggerTrapKind         kind : 4;
  FoundryDebuggerWatchAccess      access : 4;
};

enum {
  PROP_0,
  PROP_ACCESS,
  PROP_INSTRUCTION_POINTER,
  PROP_DISPOSITION,
  PROP_FUNCTION,
  PROP_KIND,
  PROP_LINE,
  PROP_LINE_OFFSET,
  PROP_PATH,
  PROP_STACK_FRAME_ID,
  PROP_THREAD_ID,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryDebuggerTrapParams, foundry_debugger_trap_params, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_trap_params_finalize (GObject *object)
{
  FoundryDebuggerTrapParams *self = (FoundryDebuggerTrapParams *)object;

  g_clear_pointer (&self->path, g_free);
  g_clear_pointer (&self->function, g_free);
  g_clear_pointer (&self->thread_id, g_free);
  g_clear_pointer (&self->stack_frame_id, g_free);

  G_OBJECT_CLASS (foundry_debugger_trap_params_parent_class)->finalize (object);
}

static void
foundry_debugger_trap_params_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  FoundryDebuggerTrapParams *self = FOUNDRY_DEBUGGER_TRAP_PARAMS (object);

  switch (prop_id)
    {
    case PROP_ACCESS:
      g_value_set_flags (value, foundry_debugger_trap_params_get_access (self));
      break;

    case PROP_DISPOSITION:
      g_value_set_enum (value, foundry_debugger_trap_params_get_disposition (self));
      break;

    case PROP_FUNCTION:
      g_value_take_string (value, foundry_debugger_trap_params_dup_function (self));
      break;

    case PROP_INSTRUCTION_POINTER:
      g_value_set_uint64 (value, foundry_debugger_trap_params_get_instruction_pointer (self));
      break;

    case PROP_KIND:
      g_value_set_enum (value, foundry_debugger_trap_params_get_kind (self));
      break;

    case PROP_LINE:
      g_value_set_uint (value, foundry_debugger_trap_params_get_line (self));
      break;

    case PROP_LINE_OFFSET:
      g_value_set_uint (value, foundry_debugger_trap_params_get_line_offset (self));
      break;

    case PROP_PATH:
      g_value_take_string (value, foundry_debugger_trap_params_dup_path (self));
      break;

    case PROP_STACK_FRAME_ID:
      g_value_take_string (value, foundry_debugger_trap_params_dup_stack_frame_id (self));
      break;

    case PROP_THREAD_ID:
      g_value_take_string (value, foundry_debugger_trap_params_dup_thread_id (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_trap_params_set_property (GObject      *object,
                                           guint         prop_id,
                                           const GValue *value,
                                           GParamSpec   *pspec)
{
  FoundryDebuggerTrapParams *self = FOUNDRY_DEBUGGER_TRAP_PARAMS (object);

  switch (prop_id)
    {
    case PROP_ACCESS:
      foundry_debugger_trap_params_set_access (self, g_value_get_flags (value));
      break;

    case PROP_DISPOSITION:
      foundry_debugger_trap_params_set_disposition (self, g_value_get_enum (value));
      break;

    case PROP_FUNCTION:
      foundry_debugger_trap_params_set_function (self, g_value_get_string (value));
      break;

    case PROP_INSTRUCTION_POINTER:
      foundry_debugger_trap_params_set_instruction_pointer (self, g_value_get_uint64 (value));
      break;

    case PROP_KIND:
      foundry_debugger_trap_params_set_kind (self, g_value_get_enum (value));
      break;

    case PROP_LINE:
      foundry_debugger_trap_params_set_line (self, g_value_get_uint (value));
      break;

    case PROP_LINE_OFFSET:
      foundry_debugger_trap_params_set_line_offset (self, g_value_get_uint (value));
      break;

    case PROP_PATH:
      foundry_debugger_trap_params_set_path (self, g_value_get_string (value));
      break;

    case PROP_STACK_FRAME_ID:
      foundry_debugger_trap_params_set_stack_frame_id (self, g_value_get_string (value));
      break;

    case PROP_THREAD_ID:
      foundry_debugger_trap_params_set_thread_id (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_trap_params_class_init (FoundryDebuggerTrapParamsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_debugger_trap_params_finalize;
  object_class->get_property = foundry_debugger_trap_params_get_property;
  object_class->set_property = foundry_debugger_trap_params_set_property;

  properties[PROP_ACCESS] =
    g_param_spec_flags ("access", NULL, NULL,
                        FOUNDRY_TYPE_DEBUGGER_WATCH_ACCESS,
                        FOUNDRY_DEBUGGER_WATCH_NONE,
                        (G_PARAM_READWRITE |
                         G_PARAM_EXPLICIT_NOTIFY |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_DISPOSITION] =
    g_param_spec_enum ("disposition", NULL, NULL,
                       FOUNDRY_TYPE_DEBUGGER_TRAP_DISPOSITION,
                       FOUNDRY_DEBUGGER_TRAP_KEEP,
                       (G_PARAM_READWRITE |
                        G_PARAM_EXPLICIT_NOTIFY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_FUNCTION] =
    g_param_spec_string ("function", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_INSTRUCTION_POINTER] =
    g_param_spec_uint64 ("instruction-pointer", NULL, NULL,
                         0, G_MAXUINT64, 0,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_KIND] =
    g_param_spec_enum ("kind", NULL, NULL,
                       FOUNDRY_TYPE_DEBUGGER_TRAP_KIND,
                       FOUNDRY_DEBUGGER_TRAP_KIND_BREAKPOINT,
                       (G_PARAM_READWRITE |
                        G_PARAM_EXPLICIT_NOTIFY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_LINE] =
    g_param_spec_uint ("line", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READWRITE |
                        G_PARAM_EXPLICIT_NOTIFY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_LINE_OFFSET] =
    g_param_spec_uint ("line-offset", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READWRITE |
                        G_PARAM_EXPLICIT_NOTIFY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_PATH] =
    g_param_spec_string ("path", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_STACK_FRAME_ID] =
    g_param_spec_string ("stack-frame-id", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_THREAD_ID] =
    g_param_spec_string ("thread-id", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_trap_params_init (FoundryDebuggerTrapParams *self)
{
}

char *
foundry_debugger_trap_params_dup_path (FoundryDebuggerTrapParams *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self), NULL);

  return g_strdup (self->path);
}

guint
foundry_debugger_trap_params_get_line (FoundryDebuggerTrapParams *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self), 0);

  return self->line;
}

guint
foundry_debugger_trap_params_get_line_offset (FoundryDebuggerTrapParams *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self), 0);

  return self->line_offset;
}

FoundryDebuggerTrapDisposition
foundry_debugger_trap_params_get_disposition (FoundryDebuggerTrapParams *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self), 0);

  return self->disposition;
}

guint64
foundry_debugger_trap_params_get_instruction_pointer (FoundryDebuggerTrapParams *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self), 0);

  return self->instruction_pointer;
}

char *
foundry_debugger_trap_params_dup_function (FoundryDebuggerTrapParams *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self), NULL);

  return g_strdup (self->function);
}

char *
foundry_debugger_trap_params_dup_stack_frame_id (FoundryDebuggerTrapParams *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self), NULL);

  return g_strdup (self->stack_frame_id);
}

char *
foundry_debugger_trap_params_dup_thread_id (FoundryDebuggerTrapParams *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self), NULL);

  return g_strdup (self->thread_id);
}

void
foundry_debugger_trap_params_set_path (FoundryDebuggerTrapParams *self,
                                       const char                *path)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self));

  if (g_set_str (&self->path, path))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_PATH]);
}

void
foundry_debugger_trap_params_set_line (FoundryDebuggerTrapParams *self,
                                       guint                      line)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self));

  if (line != self->line)
    {
      self->line = line;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_LINE]);
    }
}

void
foundry_debugger_trap_params_set_line_offset (FoundryDebuggerTrapParams *self,
                                              guint                      line_offset)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self));

  if (line_offset != self->line_offset)
    {
      self->line_offset = line_offset;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_LINE_OFFSET]);
    }
}

void
foundry_debugger_trap_params_set_disposition (FoundryDebuggerTrapParams      *self,
                                              FoundryDebuggerTrapDisposition  disposition)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self));
  g_return_if_fail (disposition <= FOUNDRY_DEBUGGER_TRAP_REMOVE_NEXT_STOP);

  if (disposition != self->disposition)
    {
      self->disposition = disposition;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_DISPOSITION]);
    }
}

void
foundry_debugger_trap_params_set_stack_frame_id (FoundryDebuggerTrapParams *self,
                                                 const char                *stack_frame_id)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self));

  if (g_set_str (&self->stack_frame_id, stack_frame_id))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_STACK_FRAME_ID]);
}

void
foundry_debugger_trap_params_set_thread_id (FoundryDebuggerTrapParams *self,
                                            const char                *thread_id)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self));

  if (g_set_str (&self->thread_id, thread_id))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_THREAD_ID]);
}

void
foundry_debugger_trap_params_set_function (FoundryDebuggerTrapParams *self,
                                           const char                *function)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self));

  if (g_set_str (&self->function, function))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_FUNCTION]);
}

void
foundry_debugger_trap_params_set_instruction_pointer (FoundryDebuggerTrapParams *self,
                                                      guint64                    instruction_pointer)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self));

  if (instruction_pointer != self->instruction_pointer)
    {
      self->instruction_pointer = instruction_pointer;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_INSTRUCTION_POINTER]);
    }
}

FoundryDebuggerTrapKind
foundry_debugger_trap_params_get_kind (FoundryDebuggerTrapParams *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self), 0);

  return self->kind;
}

void
foundry_debugger_trap_params_set_kind (FoundryDebuggerTrapParams *self,
                                       FoundryDebuggerTrapKind    kind)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self));

  if (kind != self->kind)
    {
      self->kind = kind;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_KIND]);
    }
}

FoundryDebuggerWatchAccess
foundry_debugger_trap_params_get_access (FoundryDebuggerTrapParams *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self), 0);

  return self->access;
}

void
foundry_debugger_trap_params_set_access (FoundryDebuggerTrapParams  *self,
                                         FoundryDebuggerWatchAccess  access)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self));

  if (access != self->access)
    {
      self->access = access;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_KIND]);
    }
}

/**
 * foundry_debugger_trap_params_new:
 *
 * Returns: (transfer full):
 *
 * Since: 1.1
 */
FoundryDebuggerTrapParams *
foundry_debugger_trap_params_new (void)
{
  return g_object_new (FOUNDRY_TYPE_DEBUGGER_TRAP_PARAMS, NULL);
}

/**
 * foundry_debugger_trap_params_copy:
 * @self: (nullable): a [class@Foundry.DebuggerTrapParams]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryDebuggerTrapParams *
foundry_debugger_trap_params_copy (FoundryDebuggerTrapParams *self)
{
  FoundryDebuggerTrapParams *copy;

  g_return_val_if_fail (!self || FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (self), NULL);

  if (self == NULL)
    return NULL;

  copy = foundry_debugger_trap_params_new ();

  g_set_str (&copy->path, self->path);
  g_set_str (&copy->function, self->function);
  g_set_str (&copy->thread_id, self->thread_id);
  g_set_str (&copy->stack_frame_id, self->stack_frame_id);

  copy->instruction_pointer = self->instruction_pointer;
  copy->line = self->line;
  copy->line_offset = self->line_offset;
  copy->disposition = self->disposition;
  copy->kind = self->kind;
  copy->access = self->access;

  return copy;
}

G_DEFINE_ENUM_TYPE (FoundryDebuggerTrapDisposition, foundry_debugger_trap_disposition,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_TRAP_KEEP, "keep"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_TRAP_DISABLE, "disable"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_TRAP_REMOVE_NEXT_HIT, "next-hit"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_TRAP_REMOVE_NEXT_STOP, "next-stop"))

G_DEFINE_ENUM_TYPE (FoundryDebuggerTrapKind, foundry_debugger_trap_kind,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_TRAP_KIND_BREAKPOINT, "breakpoint"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_TRAP_KIND_COUNTPOINT, "countpoint"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_TRAP_KIND_WATCHPOINT, "watchpoint"))

G_DEFINE_FLAGS_TYPE (FoundryDebuggerWatchAccess, foundry_debugger_watch_access,
                     G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_WATCH_NONE, "none"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_WATCH_READ, "read"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_WATCH_WRITE, "write"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_WATCH_READWRITE, "readwrite"))
