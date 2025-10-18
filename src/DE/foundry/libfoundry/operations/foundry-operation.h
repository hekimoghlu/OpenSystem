/* foundry-operation.h
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

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_OPERATION (foundry_operation_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryOperation, foundry_operation, FOUNDRY, OPERATION, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryOperation    *foundry_operation_new               (void);
FOUNDRY_AVAILABLE_IN_ALL
char                *foundry_operation_dup_title         (FoundryOperation    *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_operation_set_title         (FoundryOperation    *self,
                                                          const char          *title);
FOUNDRY_AVAILABLE_IN_ALL
char                *foundry_operation_dup_subtitle      (FoundryOperation    *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_operation_set_subtitle      (FoundryOperation    *self,
                                                          const char          *subtitle);
FOUNDRY_AVAILABLE_IN_ALL
double               foundry_operation_get_progress      (FoundryOperation    *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_operation_set_progress      (FoundryOperation    *self,
                                                          double               progress);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_operation_cancel            (FoundryOperation    *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_operation_complete          (FoundryOperation    *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_operation_await             (FoundryOperation    *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_operation_set_auth_provider (FoundryOperation    *self,
                                                          FoundryAuthProvider *auth_provider);
FOUNDRY_AVAILABLE_IN_ALL
FoundryAuthProvider *foundry_operation_dup_auth_provider (FoundryOperation    *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_operation_file_progress     (goffset              current_num_bytes,
                                                          goffset              total_num_bytes,
                                                          gpointer             user_data);

G_END_DECLS
