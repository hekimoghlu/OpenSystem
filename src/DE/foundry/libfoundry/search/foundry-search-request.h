/* foundry-search-request.h
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

#define FOUNDRY_SEARCH_CATEGORY_FILES         "files"
#define FOUNDRY_SEARCH_CATEGORY_ACTIONS       "actions"
#define FOUNDRY_SEARCH_CATEGORY_SYMBOLS       "symbols"
#define FOUNDRY_SEARCH_CATEGORY_DOCUMENTATION "documentation"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SEARCH_REQUEST (foundry_search_request_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundrySearchRequest, foundry_search_request, FOUNDRY, SEARCH_REQUEST, FoundryContextual)

FOUNDRY_AVAILABLE_IN_ALL
FoundrySearchRequest  *foundry_search_request_new             (FoundryContext       *context,
                                                               const char           *search_text);
FOUNDRY_AVAILABLE_IN_ALL
gboolean               foundry_search_request_has_category    (FoundrySearchRequest *self,
                                                               const char           *category);
FOUNDRY_AVAILABLE_IN_ALL
char                 **foundry_search_request_dup_categories  (FoundrySearchRequest *self);
FOUNDRY_AVAILABLE_IN_ALL
void                   foundry_search_request_set_categories  (FoundrySearchRequest *self,
                                                               const char * const   *categories);
FOUNDRY_AVAILABLE_IN_ALL
char                  *foundry_search_request_dup_search_text (FoundrySearchRequest *self);
FOUNDRY_AVAILABLE_IN_ALL
void                   foundry_search_request_set_search_text (FoundrySearchRequest *self,
                                                               const char           *search_text);

G_END_DECLS
