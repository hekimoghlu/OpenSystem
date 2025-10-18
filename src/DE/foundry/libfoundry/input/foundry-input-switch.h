/* foundry-input-switch.h
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

#define FOUNDRY_TYPE_INPUT_SWITCH (foundry_input_switch_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryInputSwitch, foundry_input_switch, FOUNDRY, INPUT_SWITCH, FoundryInput)

FOUNDRY_AVAILABLE_IN_ALL
FoundryInput *foundry_input_switch_new       (const char            *title,
                                              const char            *subtitle,
                                              FoundryInputValidator *validator,
                                              gboolean               value);
FOUNDRY_AVAILABLE_IN_ALL
gboolean      foundry_input_switch_get_value (FoundryInputSwitch    *self);
FOUNDRY_AVAILABLE_IN_ALL
void          foundry_input_switch_set_value (FoundryInputSwitch    *self,
                                              gboolean               value);

G_END_DECLS
