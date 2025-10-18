/* foundry-jsonrpc-waiter-private.h
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

#define FOUNDRY_TYPE_JSONRPC_WAITER (foundry_jsonrpc_waiter_get_type())

G_DECLARE_FINAL_TYPE (FoundryJsonrpcWaiter, foundry_jsonrpc_waiter, FOUNDRY, JSONRPC_WAITER, GObject)

FoundryJsonrpcWaiter *foundry_jsonrpc_waiter_new        (JsonNode             *node,
                                                         JsonNode             *id);
void                  foundry_jsonrpc_waiter_reply      (FoundryJsonrpcWaiter *self,
                                                         JsonNode             *node);
void                  foundry_jsonrpc_waiter_reject     (FoundryJsonrpcWaiter *self,
                                                         GError               *error);
DexFuture            *foundry_jsonrpc_waiter_await      (FoundryJsonrpcWaiter *self);
DexFuture            *foundry_jsonrpc_waiter_catch      (DexFuture            *future,
                                                         gpointer              user_data);
JsonNode             *foundry_jsonrpc_waiter_get_id     (FoundryJsonrpcWaiter *self);
JsonNode             *foundry_jsonrpc_waiter_get_node   (FoundryJsonrpcWaiter *self);

G_END_DECLS
