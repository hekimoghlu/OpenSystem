/* plugin-ollama-llm-model.h
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

#include <foundry.h>
#include <json-glib/json-glib.h>

#include "plugin-ollama-client.h"

G_BEGIN_DECLS

#define PLUGIN_TYPE_OLLAMA_LLM_MODEL (plugin_ollama_llm_model_get_type())

G_DECLARE_FINAL_TYPE (PluginOllamaLlmModel, plugin_ollama_llm_model, PLUGIN, OLLAMA_LLM_MODEL, FoundryLlmModel)

PluginOllamaLlmModel *plugin_ollama_llm_model_new (FoundryContext     *context,
                                                   PluginOllamaClient *client,
                                                   JsonNode           *node);

G_END_DECLS
