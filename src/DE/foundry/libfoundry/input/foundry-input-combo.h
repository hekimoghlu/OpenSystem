/* foundry-input-combo.h
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
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_INPUT_COMBO (foundry_input_combo_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryInputCombo, foundry_input_combo, FOUNDRY, INPUT_COMBO, FoundryInput)

FOUNDRY_AVAILABLE_IN_ALL
FoundryInput       *foundry_input_combo_new          (const char            *title,
                                                      const char            *subtitle,
                                                      FoundryInputValidator *validator,
                                                      GListModel            *choices);
FOUNDRY_AVAILABLE_IN_ALL
GListModel         *foundry_input_combo_list_choices (FoundryInputCombo     *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryInputChoice *foundry_input_combo_dup_choice   (FoundryInputCombo     *self);
FOUNDRY_AVAILABLE_IN_ALL
void                foundry_input_combo_set_choice   (FoundryInputCombo     *self,
                                                      FoundryInputChoice    *choice);

G_END_DECLS
