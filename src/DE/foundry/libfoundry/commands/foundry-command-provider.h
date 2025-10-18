/* foundry-command-provider.h
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_COMMAND_PROVIDER (foundry_command_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryCommandProvider, foundry_command_provider, FOUNDRY, COMMAND_PROVIDER, FoundryContextual)

struct _FoundryCommandProviderClass
{
  FoundryContextualClass parent_class;

  char      *(*dup_name) (FoundryCommandProvider *self);
  DexFuture *(*load)     (FoundryCommandProvider *self);
  DexFuture *(*unload)   (FoundryCommandProvider *self);

  /*< private >*/
  gpointer _reserved[8];
};

FOUNDRY_AVAILABLE_IN_ALL
char *foundry_command_provider_dup_name        (FoundryCommandProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
void  foundry_command_provider_command_added   (FoundryCommandProvider *self,
                                                FoundryCommand         *command);
FOUNDRY_AVAILABLE_IN_ALL
void  foundry_command_provider_command_removed (FoundryCommandProvider *self,
                                                FoundryCommand         *command);

G_END_DECLS

