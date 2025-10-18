/* foundry-input-validator-delegate.c
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
#include "foundry-input-validator-delegate.h"

struct _FoundryInputValidatorDelegate
{
  FoundryInputValidator parent_instance;
  FoundryInputValidatorCallback callback;
  gpointer user_data;
  GDestroyNotify notify;
} FoundryInputValidatorDelegatePrivate;

G_DEFINE_FINAL_TYPE (FoundryInputValidatorDelegate, foundry_input_validator_delegate, FOUNDRY_TYPE_INPUT_VALIDATOR)

static DexFuture *
do_nothing (DexFuture *completed,
            gpointer   user_data)
{
  return dex_ref (completed);
}

static DexFuture *
foundry_input_validator_delegate_validate (FoundryInputValidator *validator,
                                           FoundryInput          *input)
{
  FoundryInputValidatorDelegate *self = (FoundryInputValidatorDelegate *)validator;

  g_assert (FOUNDRY_IS_INPUT_VALIDATOR_DELEGATE (self));
  g_assert (FOUNDRY_IS_INPUT (input));

  /* We want to keep @self alive so that self->user_data is alive for the
   * entirety of the DexFuture's liveness. Wrap in a finally block to ensure
   * that happens.
   */
  return dex_future_finally (self->callback (input, self->user_data),
                             do_nothing,
                             g_object_ref (self),
                             g_object_unref);
}

static void
foundry_input_validator_delegate_finalize (GObject *object)
{
  FoundryInputValidatorDelegate *self = (FoundryInputValidatorDelegate *)object;

  self->callback = NULL;

  if (self->notify != NULL)
    dex_scheduler_push (dex_scheduler_get_default (),
                        g_steal_pointer (&self->notify),
                        g_steal_pointer (&self->user_data));

  G_OBJECT_CLASS (foundry_input_validator_delegate_parent_class)->finalize (object);
}

static void
foundry_input_validator_delegate_class_init (FoundryInputValidatorDelegateClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryInputValidatorClass *validator_class = FOUNDRY_INPUT_VALIDATOR_CLASS (klass);

  object_class->finalize = foundry_input_validator_delegate_finalize;

  validator_class->validate = foundry_input_validator_delegate_validate;
}

static void
foundry_input_validator_delegate_init (FoundryInputValidatorDelegate *self)
{
}

FoundryInputValidator *
foundry_input_validator_delegate_new (FoundryInputValidatorCallback callback,
                                      gpointer                      user_data,
                                      GDestroyNotify                notify)
{
  FoundryInputValidatorDelegate *self;

  g_return_val_if_fail (callback != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_INPUT_VALIDATOR_DELEGATE, NULL);
  self->callback = callback;
  self->user_data = user_data;
  self->notify = notify;

  return FOUNDRY_INPUT_VALIDATOR (self);
}
