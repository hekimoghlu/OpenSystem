/* plugin-ollama-llm-provider.c
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

#include <libsoup/soup.h>

#include "plugin-ollama-client.h"
#include "plugin-ollama-llm-provider.h"

struct _PluginOllamaLlmProvider
{
  FoundryLlmProvider  parent_instance;
  SoupSession        *session;
  PluginOllamaClient *client;
};

G_DEFINE_FINAL_TYPE (PluginOllamaLlmProvider, plugin_ollama_llm_provider, FOUNDRY_TYPE_LLM_PROVIDER)

static DexFuture *
plugin_ollama_llm_provider_list_models (FoundryLlmProvider *provider)
{
  PluginOllamaLlmProvider *self = (PluginOllamaLlmProvider *)provider;

  g_assert (PLUGIN_IS_OLLAMA_LLM_PROVIDER (self));

  return plugin_ollama_client_list_models (self->client);
}

static DexFuture *
plugin_ollama_llm_provider_load (FoundryLlmProvider *provider)
{
  PluginOllamaLlmProvider *self = (PluginOllamaLlmProvider *)provider;
  g_autoptr(FoundryContext) context = NULL;

  g_assert (PLUGIN_IS_OLLAMA_LLM_PROVIDER (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->session = soup_session_new ();
  self->client = plugin_ollama_client_new (context, self->session, NULL);

  return dex_future_new_true ();
}

static DexFuture *
plugin_ollama_llm_provider_unload (FoundryLlmProvider *provider)
{
  PluginOllamaLlmProvider *self = (PluginOllamaLlmProvider *)provider;

  g_assert (PLUGIN_IS_OLLAMA_LLM_PROVIDER (self));

  g_clear_object (&self->session);
  g_clear_object (&self->client);

  return dex_future_new_true ();
}

static void
plugin_ollama_llm_provider_class_init (PluginOllamaLlmProviderClass *klass)
{
  FoundryLlmProviderClass *llm_provider_class = FOUNDRY_LLM_PROVIDER_CLASS (klass);

  llm_provider_class->load = plugin_ollama_llm_provider_load;
  llm_provider_class->unload = plugin_ollama_llm_provider_unload;
  llm_provider_class->list_models = plugin_ollama_llm_provider_list_models;
}

static void
plugin_ollama_llm_provider_init (PluginOllamaLlmProvider *self)
{
}
