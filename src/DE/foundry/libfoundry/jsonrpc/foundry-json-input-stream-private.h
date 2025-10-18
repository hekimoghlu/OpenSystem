/* foundry-json-input-stream-private.h
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

G_BEGIN_DECLS

#define FOUNDRY_TYPE_JSON_INPUT_STREAM (foundry_json_input_stream_get_type())
#define FOUNDRY_JSON_ERROR             (foundry_json_error_quark())

typedef enum _FoundryJsonError
{
  FOUNDRY_JSON_ERROR_EOF = 1,
} FoundryJsonError;

G_DECLARE_FINAL_TYPE (FoundryJsonInputStream, foundry_json_input_stream, FOUNDRY, JSON_INPUT_STREAM, GDataInputStream)

GQuark                  foundry_json_error_quark            (void) G_GNUC_CONST;
FoundryJsonInputStream *foundry_json_input_stream_new       (GInputStream           *base_stream,
                                                             gboolean                close_base_stream);
DexFuture              *foundry_json_input_stream_read_upto (FoundryJsonInputStream *self,
                                                             const char             *stop_chars,
                                                             gssize                  stop_chars_len);
DexFuture              *foundry_json_input_stream_read_http (FoundryJsonInputStream *self);

G_END_DECLS
