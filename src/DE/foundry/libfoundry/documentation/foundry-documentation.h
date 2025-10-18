/* foundry-documentation.h
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
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DOCUMENTATION                 (foundry_documentation_get_type())
#define FOUNDRY_DOCUMENTATION_ATTRIBUTE_SINCE      "since"
#define FOUNDRY_DOCUMENTATION_ATTRIBUTE_STABILITY  "stability"
#define FOUNDRY_DOCUMENTATION_ATTRIBUTE_DEPRECATED "deprecated"

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDocumentation, foundry_documentation, FOUNDRY, DOCUMENTATION, GObject)

struct _FoundryDocumentationClass
{
  GObjectClass parent_instance;

  char      *(*dup_uri)           (FoundryDocumentation *self);
  char      *(*dup_title)         (FoundryDocumentation *self);
  GIcon     *(*dup_icon)          (FoundryDocumentation *self);
  char      *(*dup_menu_title)    (FoundryDocumentation *self);
  GIcon     *(*dup_menu_icon)     (FoundryDocumentation *self);
  char      *(*dup_section_title) (FoundryDocumentation *self);
  char      *(*dup_deprecated_in) (FoundryDocumentation *self);
  char      *(*dup_since_version) (FoundryDocumentation *self);
  gboolean   (*has_children)      (FoundryDocumentation *self);
  DexFuture *(*find_parent)       (FoundryDocumentation *self);
  DexFuture *(*find_siblings)     (FoundryDocumentation *self);
  DexFuture *(*find_children)     (FoundryDocumentation *self);
  char      *(*query_attribute)   (FoundryDocumentation *self,
                                   const char           *attribute);
  gboolean   (*equal)             (FoundryDocumentation *self,
                                   FoundryDocumentation *other);

  /*< private >*/
  gpointer _reserved[9];
};

FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_documentation_find_children     (FoundryDocumentation *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_documentation_find_parent       (FoundryDocumentation *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_documentation_find_siblings     (FoundryDocumentation *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_documentation_query_attribute   (FoundryDocumentation *self,
                                                    const char           *attribute);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_documentation_dup_deprecated_in (FoundryDocumentation *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_documentation_dup_since_version (FoundryDocumentation *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_documentation_dup_uri           (FoundryDocumentation *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_documentation_dup_title         (FoundryDocumentation *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_documentation_dup_menu_title    (FoundryDocumentation *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_documentation_dup_section_title (FoundryDocumentation *self);
FOUNDRY_AVAILABLE_IN_ALL
GIcon     *foundry_documentation_dup_menu_icon     (FoundryDocumentation *self);
FOUNDRY_AVAILABLE_IN_ALL
GIcon     *foundry_documentation_dup_icon          (FoundryDocumentation *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_documentation_has_children      (FoundryDocumentation *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_documentation_equal             (FoundryDocumentation *self,
                                                    FoundryDocumentation *other);

G_END_DECLS
