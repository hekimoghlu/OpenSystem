/* plugin-ollama-llm-completion.h
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

#include "foundry-json-input-stream-private.h"

G_BEGIN_DECLS

#define PLUGIN_TYPE_OLLAMA_LLM_COMPLETION (plugin_ollama_llm_completion_get_type())

G_DECLARE_FINAL_TYPE (PluginOllamaLlmCompletion, plugin_ollama_llm_completion, PLUGIN, OLLAMA_LLM_COMPLETION, FoundryLlmCompletion)

PluginOllamaLlmCompletion *plugin_ollama_llm_completion_new (FoundryJsonInputStream *stream);

G_END_DECLS
