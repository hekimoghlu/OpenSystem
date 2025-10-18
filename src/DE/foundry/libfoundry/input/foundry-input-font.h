/* foundry-input-font.h
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

#include "foundry-input.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_INPUT_FONT (foundry_input_font_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryInputFont, foundry_input_font, FOUNDRY, INPUT_FONT, FoundryInput)

FOUNDRY_AVAILABLE_IN_ALL
FoundryInput *foundry_input_font_new           (const char            *title,
                                                const char            *subtitle,
                                                FoundryInputValidator *validator,
                                                const char            *value,
                                                gboolean               monospace);
FOUNDRY_AVAILABLE_IN_ALL
gboolean      foundry_input_font_get_monospace (FoundryInputFont      *self);
FOUNDRY_AVAILABLE_IN_ALL
char         *foundry_input_font_dup_value     (FoundryInputFont      *self);
FOUNDRY_AVAILABLE_IN_ALL
void          foundry_input_font_set_value     (FoundryInputFont      *self,
                                                const char            *value);

G_END_DECLS
