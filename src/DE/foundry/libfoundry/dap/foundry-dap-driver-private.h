/* foundry-dap-driver-private.h
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

#include "foundry-jsonrpc-driver-private.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DAP_DRIVER (foundry_dap_driver_get_type())

G_DECLARE_FINAL_TYPE (FoundryDapDriver, foundry_dap_driver, FOUNDRY, DAP_DRIVER, GObject)

FoundryDapDriver *foundry_dap_driver_new   (GIOStream           *stream,
                                            FoundryJsonrpcStyle  style);
void              foundry_dap_driver_start (FoundryDapDriver    *self);
void              foundry_dap_driver_stop  (FoundryDapDriver    *self);
DexFuture        *foundry_dap_driver_call  (FoundryDapDriver    *self,
                                            JsonNode            *message);
DexFuture        *foundry_dap_driver_send  (FoundryDapDriver    *self,
                                            JsonNode            *message);

G_END_DECLS
