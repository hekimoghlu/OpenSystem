/* foundry-extension.h
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

#include <libpeas.h>

#include "foundry-context.h"
#include "foundry-contextual.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_EXTENSION (foundry_extension_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryExtension, foundry_extension, FOUNDRY, EXTENSION, FoundryContextual)

FOUNDRY_AVAILABLE_IN_ALL
FoundryExtension *foundry_extension_new                (FoundryContext   *context,
                                                        PeasEngine       *engine,
                                                        GType             interface_type,
                                                        const char       *key,
                                                        const char       *value);
FOUNDRY_AVAILABLE_IN_ALL
PeasEngine       *foundry_extension_get_engine         (FoundryExtension *self);
FOUNDRY_AVAILABLE_IN_ALL
gpointer          foundry_extension_get_extension      (FoundryExtension *self);
FOUNDRY_AVAILABLE_IN_ALL
GType             foundry_extension_get_interface_type (FoundryExtension *self);
FOUNDRY_AVAILABLE_IN_ALL
const char       *foundry_extension_get_key            (FoundryExtension *self);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_extension_set_key            (FoundryExtension *self,
                                                        const char       *key);
FOUNDRY_AVAILABLE_IN_ALL
const char       *foundry_extension_get_value          (FoundryExtension *self);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_extension_set_value          (FoundryExtension *self,
                                                        const char       *value);

G_END_DECLS
