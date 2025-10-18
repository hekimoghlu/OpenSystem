/* plugin-sarif-diagnostic-provider.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "plugin-sarif-diagnostic-provider.h"
#include "plugin-sarif-service.h"

struct _PluginSarifDiagnosticProvider
{
  FoundryDiagnosticProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginSarifDiagnosticProvider, plugin_sarif_diagnostic_provider, FOUNDRY_TYPE_DIAGNOSTIC_PROVIDER)

static DexFuture *
plugin_sarif_diagnostic_provider_diagnose (FoundryDiagnosticProvider *provider,
                                           GFile                     *file,
                                           GBytes                    *contents,
                                           const char                *language)
{
  PluginSarifDiagnosticProvider *self = (PluginSarifDiagnosticProvider *)provider;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(PluginSarifService) service = NULL;
  g_autoptr(GListModel) diagnostics = NULL;
  g_autoptr(GListStore) matched = NULL;
  guint n_items;

  g_assert (PLUGIN_IS_SARIF_DIAGNOSTIC_PROVIDER (self));
  g_assert (!file || G_IS_FILE (file));

  if (file == NULL)
    return foundry_future_new_not_supported ();

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  service = foundry_context_dup_service_typed (context, PLUGIN_TYPE_SARIF_SERVICE);
  diagnostics = plugin_sarif_service_list_diagnostics (service);
  n_items = g_list_model_get_n_items (G_LIST_MODEL (diagnostics));
  matched = g_list_store_new (FOUNDRY_TYPE_DIAGNOSTIC);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDiagnostic) diagnostic = g_list_model_get_item (diagnostics, i);
      g_autoptr(GFile) this_file = foundry_diagnostic_dup_file (diagnostic);

      if (this_file != NULL && g_file_equal (this_file, file))
        g_list_store_append (matched, diagnostic);
    }

  return dex_future_new_take_object (g_steal_pointer (&diagnostics));
}

static DexFuture *
plugin_sarif_diagnostic_provider_list_all (FoundryDiagnosticProvider *provider)
{
  PluginSarifDiagnosticProvider *self = (PluginSarifDiagnosticProvider *)provider;
  g_autoptr(PluginSarifService) service = NULL;
  g_autoptr(FoundryContext) context = NULL;

  g_assert (PLUGIN_IS_SARIF_DIAGNOSTIC_PROVIDER (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  service = foundry_context_dup_service_typed (context, PLUGIN_TYPE_SARIF_SERVICE);

  return dex_future_new_take_object (g_object_ref (plugin_sarif_service_list_diagnostics (service)));
}

static void
plugin_sarif_diagnostic_provider_class_init (PluginSarifDiagnosticProviderClass *klass)
{
  FoundryDiagnosticProviderClass *provider_class = FOUNDRY_DIAGNOSTIC_PROVIDER_CLASS (klass);

  provider_class->diagnose = plugin_sarif_diagnostic_provider_diagnose;
  provider_class->list_all = plugin_sarif_diagnostic_provider_list_all;
}

static void
plugin_sarif_diagnostic_provider_init (PluginSarifDiagnosticProvider *self)
{
}
