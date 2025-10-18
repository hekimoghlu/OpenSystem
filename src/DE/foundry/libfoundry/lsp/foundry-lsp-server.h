/* foundry-lsp-server.h
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

#define FOUNDRY_TYPE_LSP_SERVER (foundry_lsp_server_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryLspServer, foundry_lsp_server, FOUNDRY, LSP_SERVER, FoundryContextual)

struct _FoundryLspServerClass
{
  FoundryContextualClass parent_class;

  char       *(*dup_name)          (FoundryLspServer       *self);
  char      **(*dup_languages)     (FoundryLspServer       *self);
  DexFuture  *(*prepare)           (FoundryLspServer       *self,
                                    FoundryBuildPipeline   *pipeline,
                                    FoundryProcessLauncher *launcher);
  gboolean    (*supports_language) (FoundryLspServer       *self,
                                    const char             *language_id);

  /*< private >*/
  gpointer _reserved[8];
};

FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_lsp_server_prepare           (FoundryLspServer       *self,
                                                  FoundryBuildPipeline   *pipeline,
                                                  FoundryProcessLauncher *launcher);
FOUNDRY_AVAILABLE_IN_ALL
char       *foundry_lsp_server_dup_name          (FoundryLspServer       *self);
FOUNDRY_AVAILABLE_IN_ALL
char      **foundry_lsp_server_dup_languages     (FoundryLspServer       *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean    foundry_lsp_server_supports_language (FoundryLspServer       *self,
                                                  const char             *language_id);

G_END_DECLS
