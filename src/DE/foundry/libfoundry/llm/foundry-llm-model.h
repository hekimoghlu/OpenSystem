/* foundry-llm-model.h
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

#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LLM_MODEL (foundry_llm_model_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryLlmModel, foundry_llm_model, FOUNDRY, LLM_MODEL, FoundryContextual)

struct _FoundryLlmModelClass
{
  FoundryContextualClass parent_class;

  char      *(*dup_name)   (FoundryLlmModel            *self);
  char      *(*dup_digest) (FoundryLlmModel            *self);
  DexFuture *(*complete)   (FoundryLlmModel            *self,
                            const char * const         *roles,
                            const char * const         *messages);
  DexFuture *(*chat)       (FoundryLlmModel            *self,
                            const char                 *system);
  gboolean   (*is_metered) (FoundryLlmModel            *self);

  /*< private >*/
  gpointer _reserved[11];
};

FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_llm_model_dup_name   (FoundryLlmModel            *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_llm_model_dup_digest (FoundryLlmModel            *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_llm_model_complete   (FoundryLlmModel            *self,
                                         const char * const         *roles,
                                         const char * const         *messages);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_llm_model_chat       (FoundryLlmModel            *self,
                                         const char                 *system);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_llm_model_is_metered (FoundryLlmModel            *self);

G_END_DECLS
