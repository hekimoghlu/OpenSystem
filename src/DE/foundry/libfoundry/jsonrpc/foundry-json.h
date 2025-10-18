/* foundry-json.h
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_json_parser_load_from_file   (JsonParser         *parser,
                                                  GFile              *file) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_json_parser_load_from_stream (JsonParser         *parser,
                                                  GInputStream       *stream) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
const char *foundry_json_node_get_string_at      (JsonNode           *node,
                                                  const char         *first_key,
                                                  ...) G_GNUC_NULL_TERMINATED;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_json_node_from_bytes         (GBytes             *bytes) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_json_node_to_bytes           (JsonNode           *node) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
JsonNode   *foundry_json_node_new_strv           (const char * const *strv);

G_END_DECLS
