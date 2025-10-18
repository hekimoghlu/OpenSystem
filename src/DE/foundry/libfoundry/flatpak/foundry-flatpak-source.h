/* foundry-flatpak-source.h
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

#include "foundry-flatpak-serializable.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_FLATPAK_SOURCE (foundry_flatpak_source_get_type())

FOUNDRY_AVAILABLE_IN_ALL
FOUNDRY_DECLARE_INTERNAL_TYPE (FoundryFlatpakSource, foundry_flatpak_source, FOUNDRY, FLATPAK_SOURCE, FoundryFlatpakSerializable)

struct _FoundryFlatpakSourceClass
{
  FoundryFlatpakSerializableClass parent_class;

  /* The string "type" of the source json object */
  const char *type;

  /*< private >*/
  gpointer _reserved[7];
};

FOUNDRY_AVAILABLE_IN_ALL
char                  *foundry_flatpak_source_dup_dest        (FoundryFlatpakSource  *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
void                   foundry_flatpak_source_set_dest        (FoundryFlatpakSource  *self,
                                                               const char            *dest);
FOUNDRY_AVAILABLE_IN_ALL
char                 **foundry_flatpak_source_dup_only_arches (FoundryFlatpakSource  *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
void                   foundry_flatpak_source_set_only_arches (FoundryFlatpakSource  *self,
                                                               const char * const    *only_arches);
FOUNDRY_AVAILABLE_IN_ALL
char                 **foundry_flatpak_source_dup_skip_arches (FoundryFlatpakSource  *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
void                   foundry_flatpak_source_set_skip_arches (FoundryFlatpakSource  *self,
                                                               const char * const    *skip_arches);
FOUNDRY_AVAILABLE_IN_ALL
FoundryFlatpakSource  *foundry_flatpak_source_new_from_json   (JsonNode              *node,
                                                               GError               **error) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
