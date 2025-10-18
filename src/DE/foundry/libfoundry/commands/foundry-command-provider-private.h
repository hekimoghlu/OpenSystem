/* foundry-command-provider-private.h
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

#include "foundry-command-provider.h"

G_BEGIN_DECLS

DexFuture *foundry_command_provider_load   (FoundryCommandProvider *self);
DexFuture *foundry_command_provider_unload (FoundryCommandProvider *self);
DexFuture *foundry_command_provider_save   (FoundryCommandProvider *self);
DexFuture *foundry_command_provider_delete (FoundryCommandProvider *self,
                                           FoundryConfig         *config);
DexFuture *foundry_command_provider_copy   (FoundryCommandProvider *self,
                                           FoundryConfig         *config);

G_END_DECLS

