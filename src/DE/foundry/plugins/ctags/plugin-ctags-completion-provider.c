/* plugin-ctags-completion-provider.c
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

#include "plugin-ctags-completion-provider.h"
#include "plugin-ctags-file.h"
#include "plugin-ctags-service.h"
#include "plugin-ctags-util.h"

struct _PluginCtagsCompletionProvider
{
  FoundryCompletionProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginCtagsCompletionProvider, plugin_ctags_completion_provider, FOUNDRY_TYPE_COMPLETION_PROVIDER)

static DexFuture *
add_to_list_store (DexFuture *completed,
                   gpointer   user_data)
{
  g_autoptr(GListModel) model = dex_await_object (dex_ref (completed), NULL);
  GListStore *store = user_data;
  g_list_store_append (store, model);
  return dex_future_new_true ();
}

static DexFuture *
plugin_ctags_completion_provider_complete (FoundryCompletionProvider *provider,
                                           FoundryCompletionRequest  *request)
{
  g_autoptr(FoundryService) service = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListModel) flatten = NULL;
  g_autoptr(GListModel) files = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autofree char *word = NULL;
  g_autofree char *language_id = NULL;
  guint n_files;

  g_assert (PLUGIN_IS_CTAGS_COMPLETION_PROVIDER (provider));

  language_id = foundry_completion_request_dup_language_id (request);

  if (!plugin_ctags_is_known_language_id (language_id))
    return foundry_future_new_not_supported ();

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (provider));
  word = foundry_completion_request_dup_word (request);
  service = foundry_context_dup_service_typed (context, PLUGIN_TYPE_CTAGS_SERVICE);
  files = plugin_ctags_service_list_files (PLUGIN_CTAGS_SERVICE (service));
  n_files = g_list_model_get_n_items (files);

  if (n_files == 0)
    return foundry_future_new_not_supported ();

  store = g_list_store_new (G_TYPE_LIST_MODEL);
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_files; i++)
    {
      g_autoptr(PluginCtagsFile) file = g_list_model_get_item (files, i);

      g_ptr_array_add (futures,
                       dex_future_then (plugin_ctags_file_match (file, word),
                                        add_to_list_store,
                                        g_object_ref (store),
                                        g_object_unref));
    }

  flatten = foundry_flatten_list_model_new (g_object_ref (G_LIST_MODEL (store)));
  foundry_list_model_set_future (flatten, foundry_future_all (futures));

  return dex_future_new_take_object (g_steal_pointer (&flatten));
}

static void
plugin_ctags_completion_provider_class_init (PluginCtagsCompletionProviderClass *klass)
{
  FoundryCompletionProviderClass *provider_class = FOUNDRY_COMPLETION_PROVIDER_CLASS (klass);

  provider_class->complete = plugin_ctags_completion_provider_complete;
}

static void
plugin_ctags_completion_provider_init (PluginCtagsCompletionProvider *self)
{
}
