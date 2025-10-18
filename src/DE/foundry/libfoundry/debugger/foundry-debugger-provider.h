/* foundry-debugger-provider.h
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

#include <libpeas.h>

#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEBUGGER_PROVIDER (foundry_debugger_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDebuggerProvider, foundry_debugger_provider, FOUNDRY, DEBUGGER_PROVIDER, FoundryContextual)

struct _FoundryDebuggerProviderClass
{
  FoundryContextualClass parent_class;

  DexFuture *(*load)          (FoundryDebuggerProvider *self);
  DexFuture *(*unload)        (FoundryDebuggerProvider *self);
  DexFuture *(*supports)      (FoundryDebuggerProvider *self,
                               FoundryBuildPipeline    *pipeline,
                               FoundryCommand          *command);
  DexFuture *(*load_debugger) (FoundryDebuggerProvider *self,
                               FoundryBuildPipeline    *pipeline);

  /*< private >*/
  gpointer _reserved[8];
};

FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_debugger_provider_load_debugger   (FoundryDebuggerProvider *self,
                                                           FoundryBuildPipeline    *pipeline) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_debugger_provider_supports        (FoundryDebuggerProvider *self,
                                                           FoundryBuildPipeline    *pipeline,
                                                           FoundryCommand          *command) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_1_1
PeasPluginInfo *foundry_debugger_provider_dup_plugin_info (FoundryDebuggerProvider *self);

G_END_DECLS
