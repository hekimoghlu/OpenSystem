/* foundry-template-provider.h
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

#define FOUNDRY_TYPE_TEMPLATE_PROVIDER (foundry_template_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryTemplateProvider, foundry_template_provider, FOUNDRY, TEMPLATE_PROVIDER, GObject)

struct _FoundryTemplateProviderClass
{
  GObjectClass parent_class;

  DexFuture *(*load)                   (FoundryTemplateProvider *self);
  DexFuture *(*unload)                 (FoundryTemplateProvider *self);
  DexFuture *(*list_project_templates) (FoundryTemplateProvider *self);
  DexFuture *(*list_code_templates)    (FoundryTemplateProvider *self,
                                        FoundryContext          *context);

  /*< private >*/
  gpointer _reserved[11];
};

FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_template_provider_list_project_templates (FoundryTemplateProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_template_provider_list_code_templates    (FoundryTemplateProvider *self,
                                                             FoundryContext          *context);

G_END_DECLS
