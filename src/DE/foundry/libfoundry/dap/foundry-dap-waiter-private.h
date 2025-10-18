/* foundry-dap-waiter-private.h
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

#define FOUNDRY_TYPE_DAP_WAITER (foundry_dap_waiter_get_type())

G_DECLARE_FINAL_TYPE (FoundryDapWaiter, foundry_dap_waiter, FOUNDRY, DAP_WAITER, GObject)

FoundryDapWaiter *foundry_dap_waiter_new        (JsonNode         *node,
                                                 gint64            seq);
void              foundry_dap_waiter_reply      (FoundryDapWaiter *self,
                                                 JsonNode         *node);
void              foundry_dap_waiter_reject     (FoundryDapWaiter *self,
                                                 GError           *error);
DexFuture        *foundry_dap_waiter_await      (FoundryDapWaiter *self);
DexFuture        *foundry_dap_waiter_catch      (DexFuture        *future,
                                                 gpointer          user_data);
gint64            foundry_dap_waiter_get_seq    (FoundryDapWaiter *self);
JsonNode         *foundry_dap_waiter_get_node   (FoundryDapWaiter *self);

G_END_DECLS
