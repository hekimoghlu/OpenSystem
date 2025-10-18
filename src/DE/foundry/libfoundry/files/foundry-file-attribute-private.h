/* foundry-file-attribute-private.h
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

#include <gom/gom.h>

G_BEGIN_DECLS

#define FOUNDRY_TYPE_FILE_ATTRIBUTE (foundry_file_attribute_get_type())

G_DECLARE_FINAL_TYPE (FoundryFileAttribute, foundry_file_attribute, FOUNDRY, FILE_ATTRIBUTE, GomResource)

FoundryFileAttribute *foundry_file_attribute_new               (void);
char                 *foundry_file_attribute_dup_uri           (FoundryFileAttribute *self);
void                  foundry_file_attribute_set_uri           (FoundryFileAttribute *self,
                                                                const char           *uri);
char                 *foundry_file_attribute_dup_key           (FoundryFileAttribute *self);
void                  foundry_file_attribute_set_key           (FoundryFileAttribute *self,
                                                                const char           *key);
GBytes               *foundry_file_attribute_dup_value         (FoundryFileAttribute *self);
gboolean              foundry_file_attribute_get_value_boolean (FoundryFileAttribute *self);
char                 *foundry_file_attribute_dup_value_string  (FoundryFileAttribute *self);
void                  foundry_file_attribute_set_value         (FoundryFileAttribute *self,
                                                                GBytes               *value);
void                  foundry_file_attribute_set_value_boolean (FoundryFileAttribute *self,
                                                                gboolean              value);
void                  foundry_file_attribute_set_value_string  (FoundryFileAttribute *self,
                                                                const char           *value);
void                  foundry_file_attribute_apply_to          (FoundryFileAttribute *self,
                                                                GFileInfo            *file_info);
void                  foundry_file_attribute_apply_from        (FoundryFileAttribute *self,
                                                                GFileInfo            *file_info);

G_END_DECLS
