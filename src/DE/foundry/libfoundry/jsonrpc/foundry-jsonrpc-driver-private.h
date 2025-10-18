/* foundry-jsonrpc-driver-private.h
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

G_BEGIN_DECLS

typedef enum _FoundryJsonrpcStyle
{
  FOUNDRY_JSONRPC_STYLE_HTTP = 1,
  FOUNDRY_JSONRPC_STYLE_LF   = 2,
  FOUNDRY_JSONRPC_STYLE_NIL  = 3,
} FoundryJsonrpcStyle;

#define FOUNDRY_TYPE_JSONRPC_DRIVER (foundry_jsonrpc_driver_get_type())

G_DECLARE_FINAL_TYPE (FoundryJsonrpcDriver, foundry_jsonrpc_driver, FOUNDRY, JSONRPC_DRIVER, GObject)

FoundryJsonrpcDriver *foundry_jsonrpc_driver_new              (GIOStream            *stream,
                                                               FoundryJsonrpcStyle   style);
void                  foundry_jsonrpc_driver_start            (FoundryJsonrpcDriver *self);
void                  foundry_jsonrpc_driver_stop             (FoundryJsonrpcDriver *self);
DexFuture            *foundry_jsonrpc_driver_call             (FoundryJsonrpcDriver *self,
                                                               const char           *method,
                                                               JsonNode             *params);
DexFuture            *foundry_jsonrpc_driver_reply            (FoundryJsonrpcDriver *self,
                                                               JsonNode             *id,
                                                               JsonNode             *reply);
DexFuture            *foundry_jsonrpc_driver_reply_with_error (FoundryJsonrpcDriver *self,
                                                               JsonNode             *id,
                                                               int                   code,
                                                               const char           *message);
DexFuture            *foundry_jsonrpc_driver_notify           (FoundryJsonrpcDriver *self,
                                                               const char           *method,
                                                               JsonNode             *params);

G_END_DECLS
