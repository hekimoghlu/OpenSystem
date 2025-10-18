/* foundry-text-document-addin.h
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
#include "foundry-types.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TEXT_DOCUMENT_ADDIN (foundry_text_document_addin_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryTextDocumentAddin, foundry_text_document_addin, FOUNDRY, TEXT_DOCUMENT_ADDIN, FoundryContextual)

struct _FoundryTextDocumentAddinClass
{
  FoundryContextualClass parent_class;

  DexFuture *(*load)              (FoundryTextDocumentAddin *self);
  DexFuture *(*unload)            (FoundryTextDocumentAddin *self);

  DexFuture *(*pre_load)          (FoundryTextDocumentAddin *self);
  DexFuture *(*post_load)         (FoundryTextDocumentAddin *self);
  DexFuture *(*pre_save)          (FoundryTextDocumentAddin *self);
  DexFuture *(*post_save)         (FoundryTextDocumentAddin *self);

  DexFuture *(*list_code_actions) (FoundryTextDocumentAddin *self);

  /*< private >*/
  gpointer _reserved[9];
};

FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_text_document_addin_load              (FoundryTextDocumentAddin *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_text_document_addin_unload            (FoundryTextDocumentAddin *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
FoundryTextDocument *foundry_text_document_addin_dup_document      (FoundryTextDocumentAddin *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_text_document_addin_pre_save          (FoundryTextDocumentAddin *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_text_document_addin_post_save         (FoundryTextDocumentAddin *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_text_document_addin_pre_load          (FoundryTextDocumentAddin *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_text_document_addin_post_load         (FoundryTextDocumentAddin *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_text_document_addin_list_code_actions (FoundryTextDocumentAddin *self) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
