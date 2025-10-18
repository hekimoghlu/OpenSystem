/* foundry-llm-tool.h
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

#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LLM_TOOL (foundry_llm_tool_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryLlmTool, foundry_llm_tool, FOUNDRY, LLM_TOOL, FoundryContextual)

struct _FoundryLlmToolClass
{
  FoundryContextualClass parent_class;

  char        *(*dup_name)        (FoundryLlmTool *self);
  char        *(*dup_description) (FoundryLlmTool *self);
  GParamSpec **(*list_parameters) (FoundryLlmTool *self,
                                   guint          *n_parameters);
  DexFuture   *(*call)            (FoundryLlmTool *self,
                                   const GValue   *arguments,
                                   guint           n_arguments);

  /*< private >*/
  gpointer _parameters;
  gpointer _reserved[11];
};

FOUNDRY_AVAILABLE_IN_ALL
char        *foundry_llm_tool_dup_name            (FoundryLlmTool      *self);
FOUNDRY_AVAILABLE_IN_ALL
char        *foundry_llm_tool_dup_description     (FoundryLlmTool      *self);
FOUNDRY_AVAILABLE_IN_ALL
GParamSpec **foundry_llm_tool_list_parameters     (FoundryLlmTool      *self,
                                                   guint               *n_parameters);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture   *foundry_llm_tool_call                (FoundryLlmTool      *self,
                                                   const GValue        *arguments,
                                                   guint                n_arguments);
FOUNDRY_AVAILABLE_IN_ALL
void         foundry_llm_tool_class_add_parameter (FoundryLlmToolClass *tool_class,
                                                   GParamSpec          *pspec);

G_END_DECLS
