/* foundry-completion-proposal.h
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

#define FOUNDRY_TYPE_COMPLETION_PROPOSAL (foundry_completion_proposal_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryCompletionProposal, foundry_completion_proposal, FOUNDRY, COMPLETION_PROPOSAL, GObject)

struct _FoundryCompletionProposalClass
{
  GObjectClass parent_class;

  GIcon *(*dup_icon)         (FoundryCompletionProposal *self);
  char  *(*dup_typed_text)   (FoundryCompletionProposal *self);
  char  *(*dup_snippet_text) (FoundryCompletionProposal *self);
  char  *(*dup_after)        (FoundryCompletionProposal *self);
  char  *(*dup_comment)      (FoundryCompletionProposal *self);
  char  *(*dup_details)      (FoundryCompletionProposal *self);

  /*< private >*/
  gpointer _reserved[9];
};

FOUNDRY_AVAILABLE_IN_ALL
char  *foundry_completion_proposal_dup_after        (FoundryCompletionProposal *self);
FOUNDRY_AVAILABLE_IN_ALL
char  *foundry_completion_proposal_dup_details      (FoundryCompletionProposal *self);
FOUNDRY_AVAILABLE_IN_ALL
char  *foundry_completion_proposal_dup_typed_text   (FoundryCompletionProposal *self);
FOUNDRY_AVAILABLE_IN_ALL
char  *foundry_completion_proposal_dup_comment      (FoundryCompletionProposal *self);
FOUNDRY_AVAILABLE_IN_ALL
GIcon *foundry_completion_proposal_dup_icon         (FoundryCompletionProposal *self);
FOUNDRY_AVAILABLE_IN_ALL
char  *foundry_completion_proposal_dup_snippet_text (FoundryCompletionProposal *self);

G_END_DECLS
