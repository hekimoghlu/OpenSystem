/* foundry-llm-completion.c
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

#include "foundry-llm-completion.h"
#include "foundry-util.h"

G_DEFINE_ABSTRACT_TYPE (FoundryLlmCompletion, foundry_llm_completion, G_TYPE_OBJECT)

static void
foundry_llm_completion_class_init (FoundryLlmCompletionClass *klass)
{
}

static void
foundry_llm_completion_init (FoundryLlmCompletion *self)
{
}

/**
 * foundry_llm_completion_when_finished:
 * @self: a [class@Foundry.LlmCompletion]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value when the LLM completion has finished generating.
 */
DexFuture *
foundry_llm_completion_when_finished (FoundryLlmCompletion *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_LLM_COMPLETION (self));

  if (FOUNDRY_LLM_COMPLETION_GET_CLASS (self)->when_finished)
    return FOUNDRY_LLM_COMPLETION_GET_CLASS (self)->when_finished (self);

  return dex_future_new_true ();
}

/**
 * foundry_llm_completion_next_chunk:
 * @self: a [class@Foundry.LlmCompletion]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [class@Foundry.LlmCompletionChunk] or rejects with error.
 */
DexFuture *
foundry_llm_completion_next_chunk (FoundryLlmCompletion *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_LLM_COMPLETION (self));

  if (FOUNDRY_LLM_COMPLETION_GET_CLASS (self)->next_chunk)
    return FOUNDRY_LLM_COMPLETION_GET_CLASS (self)->next_chunk (self);

  return foundry_future_new_not_supported ();
}
