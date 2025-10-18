/* plugin-hover-bridge-hover-provider.c
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

#include "plugin-hover-bridge-hover-provider.h"

struct _PluginHoverBridgeHoverProvider
{
  FoundryHoverProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginHoverBridgeHoverProvider, plugin_hover_bridge_hover_provider, FOUNDRY_TYPE_HOVER_PROVIDER)

typedef struct _Populate
{
  FoundryTextDocument *document;
  guint line;
  guint line_offset;
} Populate;

static void
populate_free (Populate *state)
{
  g_clear_object (&state->document);
  g_free (state);
}

static DexFuture *
plugin_hover_bridge_hover_provider_populate_fiber (gpointer data)
{
  Populate *state = data;
  g_autoptr(GListModel) diagnostics = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GError) error = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_TEXT_DOCUMENT (state->document));

  if (!(diagnostics = dex_await_object (foundry_text_document_diagnose (state->document), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  n_items = g_list_model_get_n_items (diagnostics);
  store = g_list_store_new (FOUNDRY_TYPE_MARKUP);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDiagnostic) diagnostic = g_list_model_get_item (diagnostics, i);

      if (state->line == foundry_diagnostic_get_line (diagnostic))
        {
          g_autoptr(FoundryMarkup) markup = foundry_diagnostic_dup_markup (diagnostic);

          if (markup == NULL)
            {
              g_autofree char *message = foundry_diagnostic_dup_message (diagnostic);
              markup = foundry_markup_new_plaintext (message);
            }

          if (markup != NULL)
            g_list_store_append (store, markup);
        }
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static DexFuture *
plugin_hover_bridge_hover_provider_populate (FoundryHoverProvider  *provider,
                                             const FoundryTextIter *iter)
{
  g_autoptr(FoundryTextDocument) document = NULL;
  Populate *state;

  g_assert (PLUGIN_IS_HOVER_BRIDGE_HOVER_PROVIDER (provider));
  g_assert (iter != NULL);

  state = g_new0 (Populate, 1);
  state->document = foundry_hover_provider_dup_document (provider);
  state->line = foundry_text_iter_get_line (iter);
  state->line_offset = foundry_text_iter_get_line_offset (iter);

  return dex_scheduler_spawn (NULL, 0,
                              plugin_hover_bridge_hover_provider_populate_fiber,
                              state,
                              (GDestroyNotify) populate_free);
}

static void
plugin_hover_bridge_hover_provider_class_init (PluginHoverBridgeHoverProviderClass *klass)
{
  FoundryHoverProviderClass *hover_provider_class = FOUNDRY_HOVER_PROVIDER_CLASS (klass);

  hover_provider_class->populate = plugin_hover_bridge_hover_provider_populate;
}

static void
plugin_hover_bridge_hover_provider_init (PluginHoverBridgeHoverProvider *self)
{
}
