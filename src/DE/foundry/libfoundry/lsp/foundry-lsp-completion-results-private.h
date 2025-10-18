/* foundry-lsp-completion-results-private.h
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

#include "foundry-lsp-client.h"
#include "foundry-lsp-completion-proposal-private.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LSP_COMPLETION_RESULTS (foundry_lsp_completion_results_get_type())

G_DECLARE_FINAL_TYPE (FoundryLspCompletionResults, foundry_lsp_completion_results, FOUNDRY, LSP_COMPLETION_RESULTS, GObject)

DexFuture        *foundry_lsp_completion_results_new        (FoundryLspClient             *client,
                                                             JsonNode                     *reply,
                                                             const char                   *typed_text);
FoundryLspClient *foundry_lsp_completion_results_dup_client (FoundryLspCompletionResults  *self);
void              foundry_lsp_completion_results_refilter   (FoundryLspCompletionResults  *self,
                                                             const char                   *typed_text);
void              foundry_lsp_completion_results_unlink     (FoundryLspCompletionResults  *self,
                                                             FoundryLspCompletionProposal *proposal);

G_END_DECLS
