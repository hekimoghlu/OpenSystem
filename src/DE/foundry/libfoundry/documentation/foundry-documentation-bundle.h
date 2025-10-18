/* foundry-documentation-bundle.h
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

#define FOUNDRY_TYPE_DOCUMENTATION_BUNDLE (foundry_documentation_bundle_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDocumentationBundle, foundry_documentation_bundle, FOUNDRY, DOCUMENTATION_BUNDLE, FoundryContextual)

struct _FoundryDocumentationBundleClass
{
  FoundryContextualClass parent_class;

  gboolean    (*get_installed) (FoundryDocumentationBundle *self);
  char       *(*dup_id)        (FoundryDocumentationBundle *self);
  char       *(*dup_title)     (FoundryDocumentationBundle *self);
  char       *(*dup_subtitle)  (FoundryDocumentationBundle *self);
  char      **(*dup_tags)      (FoundryDocumentationBundle *self);
  DexFuture  *(*install)       (FoundryDocumentationBundle *self,
                                FoundryOperation           *operation,
                                DexCancellable             *cancellable);

  /*< private >*/
  gpointer _reserved[10];
};

FOUNDRY_AVAILABLE_IN_ALL
gboolean    foundry_documentation_bundle_get_installed (FoundryDocumentationBundle *self);
FOUNDRY_AVAILABLE_IN_ALL
char       *foundry_documentation_bundle_dup_id        (FoundryDocumentationBundle *self);
FOUNDRY_AVAILABLE_IN_ALL
char       *foundry_documentation_bundle_dup_title     (FoundryDocumentationBundle *self);
FOUNDRY_AVAILABLE_IN_ALL
char       *foundry_documentation_bundle_dup_subtitle  (FoundryDocumentationBundle *self);
FOUNDRY_AVAILABLE_IN_ALL
char      **foundry_documentation_bundle_dup_tags      (FoundryDocumentationBundle *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_documentation_bundle_install       (FoundryDocumentationBundle *self,
                                                        FoundryOperation           *operation,
                                                        DexCancellable             *cancellable);

G_END_DECLS
