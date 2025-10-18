/* foundry-debugger-source.h
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

#include <glib-object.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEBUGGER_SOURCE (foundry_debugger_source_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDebuggerSource, foundry_debugger_source, FOUNDRY, DEBUGGER_SOURCE, GObject)

struct _FoundryDebuggerSourceClass
{
  GObjectClass parent_class;

  char *(*dup_id)   (FoundryDebuggerSource *self);
  char *(*dup_name) (FoundryDebuggerSource *self);
  char *(*dup_path) (FoundryDebuggerSource *self);

  /*< private >*/
  gpointer _reserved[12];
};

FOUNDRY_AVAILABLE_IN_ALL
char *foundry_debugger_source_dup_name (FoundryDebuggerSource *self);
FOUNDRY_AVAILABLE_IN_ALL
char *foundry_debugger_source_dup_path (FoundryDebuggerSource *self);
FOUNDRY_AVAILABLE_IN_ALL
char *foundry_debugger_source_dup_id   (FoundryDebuggerSource *self);

G_END_DECLS
