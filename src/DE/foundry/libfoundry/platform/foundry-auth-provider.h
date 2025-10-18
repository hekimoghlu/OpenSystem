/* foundry-auth-provider.h
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

#include "foundry-contextual.h"
#include "foundry-input.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_AUTH_PROVIDER (foundry_auth_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryAuthProvider, foundry_auth_provider, FOUNDRY, AUTH_PROVIDER, FoundryContextual)

struct _FoundryAuthProviderClass
{
  FoundryContextualClass parent_class;

  DexFuture *(*prompt) (FoundryAuthProvider *self,
                        FoundryInput        *input);

  /*< private >*/
  gpointer _reserved[7];
};

FOUNDRY_AVAILABLE_IN_ALL
FoundryAuthProvider *foundry_auth_provider_new_for_context (FoundryContext      *context);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_auth_provider_prompt          (FoundryAuthProvider *self,
                                                            FoundryInput        *input);

G_END_DECLS
