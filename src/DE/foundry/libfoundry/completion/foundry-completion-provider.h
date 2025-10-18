/* foundry-completion-provider.h
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
#include <libpeas.h>

#include "foundry-completion-request.h"
#include "foundry-contextual.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_COMPLETION_PROVIDER (foundry_completion_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryCompletionProvider, foundry_completion_provider, FOUNDRY, COMPLETION_PROVIDER, FoundryContextual)

struct _FoundryCompletionProviderClass
{
  FoundryContextualClass parent_class;

  DexFuture *(*complete)   (FoundryCompletionProvider *self,
                            FoundryCompletionRequest  *request);
  DexFuture *(*refilter)   (FoundryCompletionProvider *self,
                            FoundryCompletionRequest  *request,
                            GListModel                *model);
  gboolean   (*is_trigger) (FoundryCompletionProvider *self,
                            const FoundryTextIter     *iter,
                            gunichar                   ch);

  /*< private >*/
  gpointer _reserved[5];
};

FOUNDRY_AVAILABLE_IN_ALL
PeasPluginInfo      *foundry_completion_provider_get_plugin_info (FoundryCompletionProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_completion_provider_complete        (FoundryCompletionProvider *self,
                                                                  FoundryCompletionRequest  *request);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTextDocument *foundry_completion_provider_dup_document    (FoundryCompletionProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_completion_provider_refilter        (FoundryCompletionProvider *self,
                                                                  FoundryCompletionRequest  *request,
                                                                  GListModel                *model);
FOUNDRY_AVAILABLE_IN_ALL
gboolean             foundry_completion_provider_is_trigger      (FoundryCompletionProvider *self,
                                                                  const FoundryTextIter     *iter,
                                                                  gunichar                   ch);

G_END_DECLS
