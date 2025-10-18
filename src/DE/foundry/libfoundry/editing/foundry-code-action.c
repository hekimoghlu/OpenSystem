/* foundry-code-action.c
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

#include "config.h"

#include "foundry-code-action.h"
#include "foundry-util.h"

enum {
  PROP_0,
  PROP_NAME,
  PROP_KIND,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryCodeAction, foundry_code_action, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_code_action_get_property (GObject    *object,
                                  guint       prop_id,
                                  GValue     *value,
                                  GParamSpec *pspec)
{
  FoundryCodeAction *self = FOUNDRY_CODE_ACTION (object);

  switch (prop_id)
    {
    case PROP_NAME:
      g_value_take_string (value, foundry_code_action_dup_name (self));
      break;

    case PROP_KIND:
      g_value_take_string (value, foundry_code_action_dup_kind (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_code_action_class_init (FoundryCodeActionClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_code_action_get_property;

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_KIND] =
    g_param_spec_string ("kind", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_code_action_init (FoundryCodeAction *self)
{
}

char *
foundry_code_action_dup_name (FoundryCodeAction *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CODE_ACTION (self), NULL);

  return FOUNDRY_CODE_ACTION_GET_CLASS (self)->dup_name (self);
}

char *
foundry_code_action_dup_kind (FoundryCodeAction *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CODE_ACTION (self), NULL);

  return FOUNDRY_CODE_ACTION_GET_CLASS (self)->dup_kind (self);
}

/**
 * foundry_code_action_run:
 * @self: a #FoundryCodeAction
 *
 * Runs the code action and returns a [class@Dex.Future] that resolves
 * when the action has completed.
 *
 * Returns: (transfer full): a [class@Dex.Future].
 */
DexFuture *
foundry_code_action_run (FoundryCodeAction *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_CODE_ACTION (self));

  if (FOUNDRY_CODE_ACTION_GET_CLASS (self)->run)
    return FOUNDRY_CODE_ACTION_GET_CLASS (self)->run (self);

  return foundry_future_new_not_supported ();
}
