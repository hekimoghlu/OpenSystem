/* foundry-debugger-trap.c
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

#include "foundry-debugger-trap.h"
#include "foundry-types.h"
#include "foundry-util.h"

enum {
  PROP_0,
  PROP_ID,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryDebuggerTrap, foundry_debugger_trap, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_trap_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundryDebuggerTrap *self = FOUNDRY_DEBUGGER_TRAP (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_take_string (value, foundry_debugger_trap_dup_id (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_trap_class_init (FoundryDebuggerTrapClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_trap_get_property;

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_trap_init (FoundryDebuggerTrap *self)
{
}

/**
 * foundry_debugger_trap_dup_id:
 * @self: a [class@Foundry.DebuggerTrap]
 *
 * Gets the identifier for this trap (such as breakpoint number).
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_debugger_trap_dup_id (FoundryDebuggerTrap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP (self), NULL);

  if (FOUNDRY_DEBUGGER_TRAP_GET_CLASS (self)->dup_id)
    return FOUNDRY_DEBUGGER_TRAP_GET_CLASS (self)->dup_id (self);

  return NULL;
}

gboolean
foundry_debugger_trap_is_armed (FoundryDebuggerTrap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP (self), FALSE);

  if (FOUNDRY_DEBUGGER_TRAP_GET_CLASS (self)->is_armed)
    return FOUNDRY_DEBUGGER_TRAP_GET_CLASS (self)->is_armed (self);

  return TRUE;
}

/**
 * foundry_debugger_trap_arm:
 * @self: a [class@Foundry.DebuggerTrap]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   to any value or rejects with error
 */
DexFuture *
foundry_debugger_trap_arm (FoundryDebuggerTrap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP (self), FALSE);

  if (FOUNDRY_DEBUGGER_TRAP_GET_CLASS (self)->arm)
    return FOUNDRY_DEBUGGER_TRAP_GET_CLASS (self)->arm (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_trap_disarm:
 * @self: a [class@Foundry.DebuggerTrap]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   to any value or rejects with error
 */
DexFuture *
foundry_debugger_trap_disarm (FoundryDebuggerTrap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP (self), FALSE);

  if (FOUNDRY_DEBUGGER_TRAP_GET_CLASS (self)->disarm)
    return FOUNDRY_DEBUGGER_TRAP_GET_CLASS (self)->disarm (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_trap_remove:
 * @self: a [class@Foundry.DebuggerTrap]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   to any value or rejects with error
 */
DexFuture *
foundry_debugger_trap_remove (FoundryDebuggerTrap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TRAP (self), FALSE);

  if (FOUNDRY_DEBUGGER_TRAP_GET_CLASS (self)->remove)
    return FOUNDRY_DEBUGGER_TRAP_GET_CLASS (self)->remove (self);

  return foundry_future_new_not_supported ();
}
