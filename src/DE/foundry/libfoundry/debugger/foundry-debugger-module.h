/* foundry-debugger-module.h
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

#include <gio/gio.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEBUGGER_MODULE (foundry_debugger_module_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDebuggerModule, foundry_debugger_module, FOUNDRY, DEBUGGER_MODULE, GObject)

struct _FoundryDebuggerModuleClass
{
  GObjectClass parent_class;

  char       *(*dup_id)             (FoundryDebuggerModule *self);
  GListModel *(*list_address_space) (FoundryDebuggerModule *self);
  char       *(*dup_path)           (FoundryDebuggerModule *self);
  char       *(*dup_host_path)      (FoundryDebuggerModule *self);
  char       *(*dup_name)           (FoundryDebuggerModule *self);

  /*< private >*/
  gpointer _reserved[10];
};

FOUNDRY_AVAILABLE_IN_ALL
char       *foundry_debugger_module_dup_id             (FoundryDebuggerModule *self);
FOUNDRY_AVAILABLE_IN_1_1
char       *foundry_debugger_module_dup_name           (FoundryDebuggerModule *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel *foundry_debugger_module_list_address_space (FoundryDebuggerModule *self);
FOUNDRY_AVAILABLE_IN_ALL
char       *foundry_debugger_module_dup_path           (FoundryDebuggerModule *self);
FOUNDRY_AVAILABLE_IN_ALL
char       *foundry_debugger_module_dup_host_path      (FoundryDebuggerModule *self);

G_END_DECLS
