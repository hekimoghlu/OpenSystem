/* plugin-no-sdk-provider.c
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

#include "plugin-no-sdk.h"
#include "plugin-no-sdk-provider.h"

struct _PluginNoSdkProvider
{
  FoundrySdkProvider  parent_instance;
  FoundrySdk         *sdk;
};

G_DEFINE_FINAL_TYPE (PluginNoSdkProvider, plugin_no_sdk_provider, FOUNDRY_TYPE_SDK_PROVIDER)

static DexFuture *
plugin_no_sdk_provider_load (FoundrySdkProvider *provider)
{
  PluginNoSdkProvider *self = (PluginNoSdkProvider *)provider;
  g_autoptr(FoundryContext) context = NULL;

  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_NO_SDK_PROVIDER (self));

  if ((context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    {
      self->sdk = plugin_no_sdk_new (context);
      foundry_sdk_provider_sdk_added (FOUNDRY_SDK_PROVIDER (self), self->sdk);
    }

  FOUNDRY_RETURN (dex_future_new_true ());
}

static DexFuture *
plugin_no_sdk_provider_unload (FoundrySdkProvider *provider)
{
  PluginNoSdkProvider *self = (PluginNoSdkProvider *)provider;
  g_autoptr(FoundryContext) context = NULL;

  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_NO_SDK_PROVIDER (self));

  if (self->sdk != NULL)
    {
      foundry_sdk_provider_sdk_removed (FOUNDRY_SDK_PROVIDER (self), self->sdk);
      g_clear_object (&self->sdk);
    }

  FOUNDRY_RETURN (dex_future_new_true ());
}

static void
plugin_no_sdk_provider_finalize (GObject *object)
{
  PluginNoSdkProvider *self = (PluginNoSdkProvider *)object;

  g_clear_object (&self->sdk);

  G_OBJECT_CLASS (plugin_no_sdk_provider_parent_class)->finalize (object);
}

static void
plugin_no_sdk_provider_class_init (PluginNoSdkProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundrySdkProviderClass *sdk_provider_class = FOUNDRY_SDK_PROVIDER_CLASS (klass);

  object_class->finalize = plugin_no_sdk_provider_finalize;

  sdk_provider_class->load = plugin_no_sdk_provider_load;
  sdk_provider_class->unload = plugin_no_sdk_provider_unload;
}

static void
plugin_no_sdk_provider_init (PluginNoSdkProvider *self)
{
}
