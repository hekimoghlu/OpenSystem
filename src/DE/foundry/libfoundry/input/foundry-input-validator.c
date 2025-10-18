/* foundry-input-validator.c
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

#include "config.h"

#include "foundry-input.h"
#include "foundry-input-validator.h"

G_DEFINE_ABSTRACT_TYPE (FoundryInputValidator, foundry_input_validator, G_TYPE_OBJECT)

static void
foundry_input_validator_class_init (FoundryInputValidatorClass *klass)
{
}

static void
foundry_input_validator_init (FoundryInputValidator *self)
{
}

/**
 * foundry_input_validator_validate:
 * @self: a [class@Foundry.InputValidator]
 * @input: a [class@Foundry.Input]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error.
 */
DexFuture *
foundry_input_validator_validate (FoundryInputValidator *self,
                                  FoundryInput          *input)
{
  dex_return_error_if_fail (FOUNDRY_IS_INPUT_VALIDATOR (self));
  dex_return_error_if_fail (FOUNDRY_IS_INPUT (input));

  if (FOUNDRY_INPUT_VALIDATOR_GET_CLASS (self)->validate)
    return FOUNDRY_INPUT_VALIDATOR_GET_CLASS (self)->validate (self, input);

  return dex_future_new_true ();
}
