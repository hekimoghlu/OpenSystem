/* foundry-test-provider.c
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

#include "foundry-test-provider-private.h"
#include "foundry-test.h"

G_DEFINE_ABSTRACT_TYPE (FoundryTestProvider, foundry_test_provider, FOUNDRY_TYPE_CONTEXTUAL)

static DexFuture *
foundry_test_provider_real_load (FoundryTestProvider *self)
{
  return dex_future_new_true ();
}

static DexFuture *
foundry_test_provider_real_unload (FoundryTestProvider *self)
{
  return dex_future_new_true ();
}

static void
foundry_test_provider_class_init (FoundryTestProviderClass *klass)
{
  klass->load = foundry_test_provider_real_load;
  klass->unload = foundry_test_provider_real_unload;
}

static void
foundry_test_provider_init (FoundryTestProvider *self)
{
}

DexFuture *
foundry_test_provider_load (FoundryTestProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEST_PROVIDER (self), NULL);

  return FOUNDRY_TEST_PROVIDER_GET_CLASS (self)->load (self);
}

DexFuture *
foundry_test_provider_unload (FoundryTestProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEST_PROVIDER (self), NULL);

  return FOUNDRY_TEST_PROVIDER_GET_CLASS (self)->unload (self);
}

/**
 * foundry_test_provider_list_tests:
 * @self: a [class@Foundry.TestProvider]
 *
 * List the available tests as a [iface@Gio.ListModel].
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] of [class@Foundry.Test].
 */
DexFuture *
foundry_test_provider_list_tests (FoundryTestProvider *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEST_PROVIDER (self));

  if (FOUNDRY_TEST_PROVIDER_GET_CLASS (self)->list_tests)
    return FOUNDRY_TEST_PROVIDER_GET_CLASS (self)->list_tests (self);

  return dex_future_new_take_object (g_list_store_new (FOUNDRY_TYPE_TEST));
}
