/* foundry-template-locator-private.h
 *
 * Copyright 2022-2025 Christian Hergert <chergert@redhat.com>
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

#include <tmpl-glib.h>

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TEMPLATE_LOCATOR (foundry_template_locator_get_type())

G_DECLARE_FINAL_TYPE (FoundryTemplateLocator, foundry_template_locator, FOUNDRY, TEMPLATE_LOCATOR, TmplTemplateLocator)

TmplTemplateLocator *foundry_template_locator_new              (void);
GBytes              *foundry_template_locator_dup_license_text (FoundryTemplateLocator *self);
void                 foundry_template_locator_set_license_text (FoundryTemplateLocator *self,
                                                                GBytes                 *license_text);

G_END_DECLS
