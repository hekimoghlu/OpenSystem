/* foundry-plugin-lsp-provider.c
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

#include "foundry-plugin-lsp-provider.h"
#include "foundry-plugin-lsp-server-private.h"

struct _FoundryPluginLspProvider
{
  FoundryLspProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (FoundryPluginLspProvider, foundry_plugin_lsp_provider, FOUNDRY_TYPE_LSP_PROVIDER)

static DexFuture *
foundry_plugin_lsp_provider_load (FoundryLspProvider *lsp_provider)
{
  g_autoptr(FoundryLspServer) server = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(PeasPluginInfo) plugin_info = NULL;

  g_assert (FOUNDRY_IS_PLUGIN_LSP_PROVIDER (lsp_provider));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (lsp_provider));
  plugin_info = foundry_lsp_provider_dup_plugin_info (lsp_provider);

  dex_return_error_if_fail (PEAS_IS_PLUGIN_INFO (plugin_info));
  dex_return_error_if_fail (FOUNDRY_IS_CONTEXT (context));

  server = foundry_plugin_lsp_server_new (context, plugin_info);
  foundry_lsp_provider_set_server (lsp_provider, server);

  return dex_future_new_true ();
}

static DexFuture *
foundry_plugin_lsp_provider_unload (FoundryLspProvider *lsp_provider)
{
  g_assert (FOUNDRY_IS_PLUGIN_LSP_PROVIDER (lsp_provider));

  foundry_lsp_provider_set_server (lsp_provider, NULL);

  return dex_future_new_true ();
}

static void
foundry_plugin_lsp_provider_class_init (FoundryPluginLspProviderClass *klass)
{
  FoundryLspProviderClass *lsp_provider_class = FOUNDRY_LSP_PROVIDER_CLASS (klass);

  lsp_provider_class->load = foundry_plugin_lsp_provider_load;
  lsp_provider_class->unload = foundry_plugin_lsp_provider_unload;
}

static void
foundry_plugin_lsp_provider_init (FoundryPluginLspProvider *self)
{
}
