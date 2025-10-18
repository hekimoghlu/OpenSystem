/* foundry-lsp-manager.h
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
#include <libpeas.h>

#include "foundry-service.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LSP_MANAGER (foundry_lsp_manager_get_type())

FOUNDRY_AVAILABLE_IN_ALL
FOUNDRY_DECLARE_INTERNAL_TYPE (FoundryLspManager, foundry_lsp_manager, FOUNDRY, LSP_MANAGER, FoundryService)

FOUNDRY_AVAILABLE_IN_ALL
DexFuture       *foundry_lsp_manager_load_client            (FoundryLspManager *self,
                                                             const char        *language_id) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
FoundrySettings *foundry_lsp_manager_load_language_settings (FoundryLspManager *self,
                                                             const char        *language_id);

G_END_DECLS
