/* foundry-input-validator-delegate.h
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

#include "foundry-input-validator.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_INPUT_VALIDATOR_DELEGATE (foundry_input_validator_delegate_get_type())

typedef DexFuture *(*FoundryInputValidatorCallback) (FoundryInput *input,
                                                     gpointer      user_data);

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryInputValidatorDelegate, foundry_input_validator_delegate, FOUNDRY, INPUT_VALIDATOR_DELEGATE, FoundryInputValidator)

FOUNDRY_AVAILABLE_IN_ALL
FoundryInputValidator *foundry_input_validator_delegate_new (FoundryInputValidatorCallback callback,
                                                             gpointer                      user_data,
                                                             GDestroyNotify                notify);

G_END_DECLS
