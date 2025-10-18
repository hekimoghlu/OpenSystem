/* foundry-llm-completion.h
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

#include <libdex.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LLM_COMPLETION (foundry_llm_completion_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryLlmCompletion, foundry_llm_completion, FOUNDRY, LLM_COMPLETION, GObject)

struct _FoundryLlmCompletionClass
{
  GObjectClass parent_class;

  DexFuture *(*when_finished) (FoundryLlmCompletion *self);
  DexFuture *(*next_chunk)    (FoundryLlmCompletion *self);

  /*< private >*/
  gpointer _reserved[13];
};

FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_llm_completion_when_finished (FoundryLlmCompletion *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_llm_completion_next_chunk    (FoundryLlmCompletion *self);

G_END_DECLS
