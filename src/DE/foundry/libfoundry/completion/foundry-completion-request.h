/* foundry-completion-request.h
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

#include "foundry-text-buffer.h"
#include "foundry-text-document.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_COMPLETION_REQUEST (foundry_completion_request_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryCompletionRequest, foundry_completion_request, FOUNDRY, COMPLETION_REQUEST, GObject)

struct _FoundryCompletionRequestClass
{
  GObjectClass parent_class;

  GFile                       *(*dup_file)        (FoundryCompletionRequest *self);
  char                        *(*dup_language_id) (FoundryCompletionRequest *self);
  char                        *(*dup_word)        (FoundryCompletionRequest *self);
  void                         (*get_bounds)      (FoundryCompletionRequest *self,
                                                   FoundryTextIter          *begin,
                                                   FoundryTextIter          *end);
  FoundryCompletionActivation  (*get_activation)  (FoundryCompletionRequest *self);

  /*< private >*/
  gpointer _reserved[10];
};

FOUNDRY_AVAILABLE_IN_ALL
GFile                       *foundry_completion_request_dup_file        (FoundryCompletionRequest *self);
FOUNDRY_AVAILABLE_IN_ALL
char                        *foundry_completion_request_dup_word        (FoundryCompletionRequest *self);
FOUNDRY_AVAILABLE_IN_ALL
char                        *foundry_completion_request_dup_language_id (FoundryCompletionRequest *self);
FOUNDRY_AVAILABLE_IN_ALL
void                         foundry_completion_request_get_bounds      (FoundryCompletionRequest *self,
                                                                         FoundryTextIter          *begin,
                                                                         FoundryTextIter          *end);
FOUNDRY_AVAILABLE_IN_ALL
FoundryCompletionActivation  foundry_completion_request_get_activation  (FoundryCompletionRequest *self);

G_END_DECLS
