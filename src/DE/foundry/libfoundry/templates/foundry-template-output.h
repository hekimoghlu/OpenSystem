/* foundry-template-output.h
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <libdex.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TEMPLATE_OUTPUT (foundry_template_output_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryTemplateOutput, foundry_template_output, FOUNDRY, TEMPLATE_OUTPUT, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryTemplateOutput *foundry_template_output_new           (GFile                 *file,
                                                              GBytes                *contents,
                                                              int                    mode);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTemplateOutput *foundry_template_output_new_directory (GFile                 *file);
FOUNDRY_AVAILABLE_IN_ALL
GBytes                *foundry_template_output_dup_contents  (FoundryTemplateOutput *self);
FOUNDRY_AVAILABLE_IN_ALL
GFile                 *foundry_template_output_dup_file      (FoundryTemplateOutput *self);
FOUNDRY_AVAILABLE_IN_ALL
int                    foundry_template_output_get_mode      (FoundryTemplateOutput *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture             *foundry_template_output_write         (FoundryTemplateOutput *self);

G_END_DECLS
