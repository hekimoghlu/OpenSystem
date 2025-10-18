/* foundry-debugger-trap.h
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

#pragma once

#include <libdex.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEBUGGER_TRAP (foundry_debugger_trap_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDebuggerTrap, foundry_debugger_trap, FOUNDRY, DEBUGGER_TRAP, GObject)

struct _FoundryDebuggerTrapClass
{
  GObjectClass parent_class;

  char      *(*dup_id)   (FoundryDebuggerTrap *self);
  gboolean   (*is_armed) (FoundryDebuggerTrap *self);
  DexFuture *(*arm)      (FoundryDebuggerTrap *self);
  DexFuture *(*disarm)   (FoundryDebuggerTrap *self);
  DexFuture *(*remove)   (FoundryDebuggerTrap *self);

  /*< private >*/
  gpointer _reserved[10];
};

FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_debugger_trap_dup_id   (FoundryDebuggerTrap *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_debugger_trap_is_armed (FoundryDebuggerTrap *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_debugger_trap_arm      (FoundryDebuggerTrap *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_debugger_trap_disarm   (FoundryDebuggerTrap *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_debugger_trap_remove   (FoundryDebuggerTrap *self);

G_END_DECLS
