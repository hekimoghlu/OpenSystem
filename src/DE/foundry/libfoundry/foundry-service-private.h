/* foundry-service-private.h
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

#include "foundry-contextual.h"
#include "foundry-service.h"

G_BEGIN_DECLS

DexFuture    *foundry_service_start            (FoundryService *self) G_GNUC_WARN_UNUSED_RESULT;
DexFuture    *foundry_service_stop             (FoundryService *self) G_GNUC_WARN_UNUSED_RESULT;
GActionGroup *foundry_service_get_action_group (FoundryService *self);

G_END_DECLS
