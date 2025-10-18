/* foundry-flatpak-serializable.h
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
#include <json-glib/json-glib.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_FLATPAK_SERIALIZABLE (foundry_flatpak_serializable_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryFlatpakSerializable, foundry_flatpak_serializable, FOUNDRY, FLATPAK_SERIALIZABLE, GObject)

struct _FoundryFlatpakSerializableClass
{
  GObjectClass parent_class;

  DexFuture *(*deserialize)          (FoundryFlatpakSerializable *self,
                                      JsonNode                  *node);
  DexFuture *(*deserialize_property) (FoundryFlatpakSerializable *self,
                                      const char                *property_name,
                                      JsonNode                  *property_node);

  /*< private >*/
  gpointer _reserved[13];
};

FOUNDRY_AVAILABLE_IN_ALL
GFile  *foundry_flatpak_serializable_resolve_file (FoundryFlatpakSerializable  *self,
                                                   const char                  *path,
                                                   GError                     **error);
FOUNDRY_AVAILABLE_IN_ALL
char   *foundry_flatpak_serializable_dup_x_string (FoundryFlatpakSerializable  *self,
                                                   const char                  *property);
FOUNDRY_AVAILABLE_IN_ALL
char  **foundry_flatpak_serializable_dup_x_strv   (FoundryFlatpakSerializable  *self,
                                                   const char                  *property);

G_END_DECLS
