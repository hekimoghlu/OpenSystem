/* plugin-word-completion-provider.c
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

#include <gtk/gtk.h>

#include "plugin-word-completion-proposal.h"
#include "plugin-word-completion-provider.h"
#include "plugin-word-completion-results.h"

struct _PluginWordCompletionProvider
{
  FoundryCompletionProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginWordCompletionProvider, plugin_word_completion_provider, FOUNDRY_TYPE_COMPLETION_PROVIDER)

static GtkExpression *expression;

static DexFuture *
plugin_word_completion_provider_complete (FoundryCompletionProvider *provider,
                                          FoundryCompletionRequest  *request)
{
  PluginWordCompletionProvider *self = (PluginWordCompletionProvider *)provider;
  g_autoptr(PluginWordCompletionResults) model = NULL;
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(GtkFilterListModel) filtered = NULL;
  g_autoptr(FoundryTextBuffer) buffer = NULL;
  g_autoptr(GtkFilter) filter = NULL;
  g_autoptr(DexFuture) future = NULL;
  g_autoptr(GBytes) bytes = NULL;
  g_autoptr(GFile) file = NULL;
  g_autofree char *language_id = NULL;
  g_autofree char *word = NULL;

  g_assert (PLUGIN_IS_WORD_COMPLETION_PROVIDER (self));
  g_assert (FOUNDRY_IS_COMPLETION_REQUEST (request));

  document = foundry_completion_provider_dup_document (provider);
  file = foundry_text_document_dup_file (document);
  buffer = foundry_text_document_dup_buffer (document);
  bytes = foundry_text_buffer_dup_contents (buffer);
  word = foundry_completion_request_dup_word (request);
  language_id = foundry_completion_request_dup_language_id (request);

  filter = g_object_new (GTK_TYPE_STRING_FILTER,
                         "expression", expression,
                         "match-mode", GTK_STRING_FILTER_MATCH_MODE_PREFIX,
                         "ignore-case", TRUE,
                         "search", word,
                         NULL);

  model = plugin_word_completion_results_new (file, bytes, language_id);
  future = plugin_word_completion_results_await (model);
  filtered = gtk_filter_list_model_new (g_object_ref (G_LIST_MODEL (model)),
                                        g_steal_pointer (&filter));

  foundry_list_model_set_future (G_LIST_MODEL (filtered), future);

  return dex_future_new_take_object (g_steal_pointer (&filtered));
}

static DexFuture *
plugin_word_completion_provider_refilter (FoundryCompletionProvider *provider,
                                          FoundryCompletionRequest  *request,
                                          GListModel                *model)
{
  g_autofree char *word = NULL;
  GtkFilter *filter;

  g_assert (PLUGIN_IS_WORD_COMPLETION_PROVIDER (provider));
  g_assert (FOUNDRY_IS_COMPLETION_REQUEST (request));
  g_assert (GTK_IS_FILTER_LIST_MODEL (model));

  filter = gtk_filter_list_model_get_filter (GTK_FILTER_LIST_MODEL (model));
  word = foundry_completion_request_dup_word (request);

  g_assert (GTK_IS_STRING_FILTER (filter));

  gtk_string_filter_set_search (GTK_STRING_FILTER (filter), word);

  return dex_future_new_take_object (g_object_ref (model));
}

static void
plugin_word_completion_provider_class_init (PluginWordCompletionProviderClass *klass)
{
  FoundryCompletionProviderClass *completion_provider_class = FOUNDRY_COMPLETION_PROVIDER_CLASS (klass);

  completion_provider_class->complete = plugin_word_completion_provider_complete;
  completion_provider_class->refilter = plugin_word_completion_provider_refilter;

  expression = gtk_property_expression_new (PLUGIN_TYPE_WORD_COMPLETION_PROPOSAL, NULL, "word");
}

static void
plugin_word_completion_provider_init (PluginWordCompletionProvider *self)
{
}
