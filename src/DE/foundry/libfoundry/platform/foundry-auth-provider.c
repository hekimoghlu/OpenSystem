/* foundry-auth-provider.c
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

#include <libpeas.h>

#include "foundry-auth-provider.h"
#include "foundry-extension.h"
#include "foundry-util.h"

G_DEFINE_ABSTRACT_TYPE (FoundryAuthProvider, foundry_auth_provider, FOUNDRY_TYPE_CONTEXTUAL)

static void
foundry_auth_provider_class_init (FoundryAuthProviderClass *klass)
{
}

static void
foundry_auth_provider_init (FoundryAuthProvider *self)
{
}

/**
 * foundry_auth_provider_prompt:
 * @self: a [class@Foundry.AuthProvider]
 * @input: a [class@Foundry.Input]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any value
 *   when the prompt has been completed by the user.
 */
DexFuture *
foundry_auth_provider_prompt (FoundryAuthProvider *self,
                              FoundryInput        *input)
{
  dex_return_error_if_fail (FOUNDRY_IS_AUTH_PROVIDER (self));
  dex_return_error_if_fail (FOUNDRY_IS_INPUT (input));

  if (FOUNDRY_AUTH_PROVIDER_GET_CLASS (self)->prompt)
    return FOUNDRY_AUTH_PROVIDER_GET_CLASS (self)->prompt (self, input);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_auth_provider_new_for_context:
 * @context: a [class@Foundry.Context]
 *
 * Creates a new [class@Foundry.AuthProvider] for @context.
 *
 * Returns: (transfer full) (nullable):
 */
FoundryAuthProvider *
foundry_auth_provider_new_for_context (FoundryContext *context)
{
  g_autoptr(FoundryExtension) adapter = NULL;
  FoundryAuthProvider *provider;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);

  adapter = foundry_extension_new (context,
                                   peas_engine_get_default (),
                                   FOUNDRY_TYPE_AUTH_PROVIDER,
                                   "Auth-Provider", "*");

  if ((provider = foundry_extension_get_extension (adapter)))
    return g_object_ref (provider);

  return NULL;
}
