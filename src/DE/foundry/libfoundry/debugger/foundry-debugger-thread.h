/* foundry-debugger-thread.h
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

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEBUGGER_THREAD (foundry_debugger_thread_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDebuggerThread, foundry_debugger_thread, FOUNDRY, DEBUGGER_THREAD, GObject)

struct _FoundryDebuggerThreadClass
{
  GObjectClass parent_class;

  char      *(*dup_id)       (FoundryDebuggerThread   *self);
  char      *(*dup_group_id) (FoundryDebuggerThread   *self);
  DexFuture *(*list_frames)  (FoundryDebuggerThread   *self);
  gboolean   (*is_stopped)   (FoundryDebuggerThread   *self);
  DexFuture *(*move)         (FoundryDebuggerThread   *self,
                              FoundryDebuggerMovement  movement);
  DexFuture *(*interrupt)    (FoundryDebuggerThread   *self);
  gboolean   (*can_move)     (FoundryDebuggerThread   *self,
                              FoundryDebuggerMovement  movement);

  /*< private >*/
  gpointer _reserved[8];
};

FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_debugger_thread_dup_id       (FoundryDebuggerThread   *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_debugger_thread_dup_group_id (FoundryDebuggerThread   *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_debugger_thread_list_frames  (FoundryDebuggerThread   *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_debugger_thread_is_stopped   (FoundryDebuggerThread   *self);
FOUNDRY_AVAILABLE_IN_1_1
DexFuture *foundry_debugger_thread_move         (FoundryDebuggerThread   *self,
                                                 FoundryDebuggerMovement  movement);
FOUNDRY_AVAILABLE_IN_1_1
gboolean   foundry_debugger_thread_can_move     (FoundryDebuggerThread   *self,
                                                 FoundryDebuggerMovement  movement);
FOUNDRY_AVAILABLE_IN_1_1
DexFuture *foundry_debugger_thread_interrupt    (FoundryDebuggerThread   *self);

G_END_DECLS
