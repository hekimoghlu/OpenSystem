/* foundry-template-manager.h
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

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TEMPLATE_MANAGER (foundry_template_manager_get_type())

FOUNDRY_AVAILABLE_IN_ALL
FOUNDRY_DECLARE_INTERNAL_TYPE (FoundryTemplateManager, foundry_template_manager, FOUNDRY, TEMPLATE_MANAGER, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryTemplateManager *foundry_template_manager_new                    (void);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture              *foundry_template_manager_list_project_templates (FoundryTemplateManager *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture              *foundry_template_manager_list_code_templates    (FoundryTemplateManager *self,
                                                                         FoundryContext         *context);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture              *foundry_template_manager_list_templates         (FoundryTemplateManager *self,
                                                                         FoundryContext         *context);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture              *foundry_template_manager_find_template          (FoundryTemplateManager *self,
                                                                         FoundryContext         *context,
                                                                         const char             *template_id);

G_END_DECLS
