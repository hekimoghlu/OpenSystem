/* foundry-lsp-completion-proposal-private.h
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

#include <json-glib/json-glib.h>

#include "foundry-lsp-completion-proposal.h"

G_BEGIN_DECLS

enum {
  LSP_COMPLETION_TEXT           = 1,
  LSP_COMPLETION_METHOD         = 2,
  LSP_COMPLETION_FUNCTION       = 3,
  LSP_COMPLETION_CONSTRUCTOR    = 4,
  LSP_COMPLETION_FIELD          = 5,
  LSP_COMPLETION_VARIABLE       = 6,
  LSP_COMPLETION_CLASS          = 7,
  LSP_COMPLETION_INTERFACE      = 8,
  LSP_COMPLETION_MODULE         = 9,
  LSP_COMPLETION_PROPERTY       = 10,
  LSP_COMPLETION_UNIT           = 11,
  LSP_COMPLETION_VALUE          = 12,
  LSP_COMPLETION_ENUM           = 13,
  LSP_COMPLETION_KEYWORD        = 14,
  LSP_COMPLETION_SNIPPET        = 15,
  LSP_COMPLETION_COLOR          = 16,
  LSP_COMPLETION_FILE           = 17,
  LSP_COMPLETION_REFERENCE      = 18,
  LSP_COMPLETION_FOLDER         = 19,
  LSP_COMPLETION_ENUM_MEMBER    = 20,
  LSP_COMPLETION_CONSTANT       = 21,
  LSP_COMPLETION_STRUCT         = 22,
  LSP_COMPLETION_EVENT          = 23,
  LSP_COMPLETION_OPERATOR       = 24,
  LSP_COMPLETION_TYPE_PARAMETER = 25,
};

struct _FoundryLspCompletionProposal
{
  FoundryCompletionProposal             parent_instance;

  gpointer                              container;
  gpointer                             *indexed;
  GList                                 link;

  JsonNode                             *info;
  const char                           *label;
  const char                           *detail;
  const char                           *after;
  guint                                 kind;
};

FoundryLspCompletionProposal *_foundry_lsp_completion_proposal_new (JsonNode *info);

G_END_DECLS
