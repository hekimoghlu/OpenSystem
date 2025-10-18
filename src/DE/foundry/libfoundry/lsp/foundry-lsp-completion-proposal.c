/* foundry-lsp-completion-proposal.c
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

#include "foundry-json-node.h"
#include "foundry-lsp-completion-proposal-private.h"
#include "foundry-lsp-completion-results-private.h"

G_DEFINE_FINAL_TYPE (FoundryLspCompletionProposal, foundry_lsp_completion_proposal, FOUNDRY_TYPE_COMPLETION_PROPOSAL)

static char *
foundry_lsp_completion_proposal_dup_typed_text (FoundryCompletionProposal *proposal)
{
  return g_strdup (FOUNDRY_LSP_COMPLETION_PROPOSAL (proposal)->label);
}

static char *
foundry_lsp_completion_proposal_dup_details (FoundryCompletionProposal *proposal)
{
  return g_strdup (FOUNDRY_LSP_COMPLETION_PROPOSAL (proposal)->detail);
}

static GIcon *
foundry_lsp_completion_proposal_dup_icon (FoundryCompletionProposal *proposal)
{
  FoundryLspCompletionProposal *self = FOUNDRY_LSP_COMPLETION_PROPOSAL (proposal);

  switch (self->kind)
    {
    case LSP_COMPLETION_METHOD:
      return g_themed_icon_new ("lang-method-symbolic");

    case LSP_COMPLETION_CONSTRUCTOR:
    case LSP_COMPLETION_FUNCTION:
      return g_themed_icon_new ("lang-function-symbolic");

    case LSP_COMPLETION_VARIABLE:
      return g_themed_icon_new ("lang-struct-field-symbolic");

    case LSP_COMPLETION_CLASS:
      return g_themed_icon_new ("lang-class-symbolic");

    case LSP_COMPLETION_PROPERTY:
      return g_themed_icon_new ("lang-property-symbolic");

    case LSP_COMPLETION_ENUM:
      return g_themed_icon_new ("lang-enum-symbolic");

    case LSP_COMPLETION_ENUM_MEMBER:
      return g_themed_icon_new ("lang-constant-symbolic");

    case LSP_COMPLETION_STRUCT:
      return g_themed_icon_new ("lang-struct-symbolic");

    default:
      break;
    }

  return NULL;
}

static void
foundry_lsp_completion_proposal_finalize (GObject *object)
{
  FoundryLspCompletionProposal *self = (FoundryLspCompletionProposal *)object;

  g_assert ((self->container && self->indexed) ||
            (!self->container && !self->indexed));

  self->label = NULL;
  self->detail = NULL;

  if (self->container != NULL)
    foundry_lsp_completion_results_unlink (self->container, self);

  g_clear_pointer (&self->info, json_node_unref);

  self->link.data = NULL;

  g_assert (self->container == NULL);
  g_assert (self->indexed == NULL);
  g_assert (self->link.prev == NULL);
  g_assert (self->link.next == NULL);
  g_assert (self->link.data == NULL);

  G_OBJECT_CLASS (foundry_lsp_completion_proposal_parent_class)->finalize (object);
}

static void
foundry_lsp_completion_proposal_class_init (FoundryLspCompletionProposalClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryCompletionProposalClass *proposal_class = FOUNDRY_COMPLETION_PROPOSAL_CLASS (klass);

  object_class->finalize = foundry_lsp_completion_proposal_finalize;

  proposal_class->dup_details = foundry_lsp_completion_proposal_dup_details;
  proposal_class->dup_icon = foundry_lsp_completion_proposal_dup_icon;
  proposal_class->dup_typed_text = foundry_lsp_completion_proposal_dup_typed_text;
}

static void
foundry_lsp_completion_proposal_init (FoundryLspCompletionProposal *self)
{
  self->link.data = self;
}

FoundryLspCompletionProposal *
_foundry_lsp_completion_proposal_new (JsonNode *info)
{
  FoundryLspCompletionProposal *self;
  const char *after = NULL;
  gint64 kind;

  g_return_val_if_fail (info != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_LSP_COMPLETION_PROPOSAL, NULL);
  self->info = json_node_ref (info);

  if (!FOUNDRY_JSON_OBJECT_PARSE (self->info, "label", FOUNDRY_JSON_NODE_GET_STRING (&self->label)))
    self->label = "";

  if (!FOUNDRY_JSON_OBJECT_PARSE (self->info, "detail", FOUNDRY_JSON_NODE_GET_STRING (&self->detail)))
    self->detail = NULL;

  while (*self->label && g_unichar_isspace (g_utf8_get_char (self->label)))
    self->label = g_utf8_next_char (self->label);

  if (FOUNDRY_JSON_OBJECT_PARSE (self->info, "kind", FOUNDRY_JSON_NODE_GET_INT (&kind)))
    self->kind = CLAMP (kind, 0, G_MAXUINT);

  if (FOUNDRY_JSON_OBJECT_PARSE (self->info,
                                 "labelDetails", "{",
                                   "detail", FOUNDRY_JSON_NODE_GET_STRING (&after),
                                 "}"))
    self->after = after;

  return self;
}
