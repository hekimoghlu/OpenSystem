/* foundry-debugger-target-process.c
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

#include "foundry-debugger-target-process.h"
#include "foundry-debugger-target-private.h"

struct _FoundryDebuggerTargetProcess
{
  FoundryDebuggerTarget parent_instance;
  GPid pid;
};

struct _FoundryDebuggerTargetProcessClass
{
  FoundryDebuggerTargetClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryDebuggerTargetProcess, foundry_debugger_target_process, FOUNDRY_TYPE_DEBUGGER_TARGET)

enum {
  PROP_0,
  PROP_PID,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_target_process_get_property (GObject    *object,
                                              guint       prop_id,
                                              GValue     *value,
                                              GParamSpec *pspec)
{
  FoundryDebuggerTargetProcess *self = FOUNDRY_DEBUGGER_TARGET_PROCESS (object);

  switch (prop_id)
    {
    case PROP_PID:
      g_value_set_int (value, foundry_debugger_target_process_get_pid (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_target_process_set_property (GObject      *object,
                                              guint         prop_id,
                                              const GValue *value,
                                              GParamSpec   *pspec)
{
  FoundryDebuggerTargetProcess *self = FOUNDRY_DEBUGGER_TARGET_PROCESS (object);

  switch (prop_id)
    {
    case PROP_PID:
      self->pid = g_value_get_int (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_target_process_class_init (FoundryDebuggerTargetProcessClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_target_process_get_property;
  object_class->set_property = foundry_debugger_target_process_set_property;

  properties[PROP_PID] =
    g_param_spec_int ("pid", NULL, NULL,
                      0, G_MAXINT, 0,
                      (G_PARAM_READWRITE |
                       G_PARAM_CONSTRUCT_ONLY |
                       G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_target_process_init (FoundryDebuggerTargetProcess *self)
{
}

/**
 * foundry_debugger_target_process_get_pid:
 * @self: a [class@Foundry.DebuggerTargetCommand]
 *
 */
GPid
foundry_debugger_target_process_get_pid (FoundryDebuggerTargetProcess *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TARGET_PROCESS (self), 0);

  return self->pid;
}
