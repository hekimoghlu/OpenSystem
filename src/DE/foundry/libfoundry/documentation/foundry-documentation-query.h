/* foundry-documentation-query.h
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

#include <glib-object.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DOCUMENTATION_QUERY (foundry_documentation_query_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryDocumentationQuery, foundry_documentation_query, FOUNDRY, DOCUMENTATION_QUERY, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryDocumentationQuery *foundry_documentation_query_new               (void);
FOUNDRY_AVAILABLE_IN_ALL
char                      *foundry_documentation_query_dup_keyword       (FoundryDocumentationQuery *self);
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_documentation_query_set_keyword       (FoundryDocumentationQuery *self,
                                                                          const char                *keyword);
FOUNDRY_AVAILABLE_IN_ALL
gboolean                   foundry_documentation_query_get_prefetch_all  (FoundryDocumentationQuery *self);
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_documentation_query_set_prefetch_all  (FoundryDocumentationQuery *self,
                                                                          gboolean                   prefetch_all);
FOUNDRY_AVAILABLE_IN_ALL
char                      *foundry_documentation_query_dup_type_name     (FoundryDocumentationQuery *self);
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_documentation_query_set_type_name     (FoundryDocumentationQuery *self,
                                                                          const char                *type_name);
FOUNDRY_AVAILABLE_IN_ALL
char                      *foundry_documentation_query_dup_property_name (FoundryDocumentationQuery *self);
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_documentation_query_set_property_name (FoundryDocumentationQuery *self,
                                                                          const char                *property_name);
FOUNDRY_AVAILABLE_IN_ALL
char                      *foundry_documentation_query_dup_function_name (FoundryDocumentationQuery *self);
FOUNDRY_AVAILABLE_IN_ALL
void                       foundry_documentation_query_set_function_name (FoundryDocumentationQuery *self,
                                                                          const char                *function_name);

G_END_DECLS
