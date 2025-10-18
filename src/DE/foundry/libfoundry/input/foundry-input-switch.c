/* foundry-input-switch.c
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

#include "foundry-input-switch.h"
#include "foundry-input-validator.h"
#include "foundry-util-private.h"

struct _FoundryInputSwitch
{
  FoundryInput parent_instance;
  guint value : 1;
};

enum {
  PROP_0,
  PROP_VALUE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryInputSwitch, foundry_input_switch, FOUNDRY_TYPE_INPUT)

static GParamSpec *properties[N_PROPS];

static void
foundry_input_switch_get_property (GObject    *object,
                                   guint       prop_id,
                                   GValue     *value,
                                   GParamSpec *pspec)
{
  FoundryInputSwitch *self = FOUNDRY_INPUT_SWITCH (object);

  switch (prop_id)
    {
    case PROP_VALUE:
      g_value_set_boolean (value, foundry_input_switch_get_value (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_switch_set_property (GObject      *object,
                                   guint         prop_id,
                                   const GValue *value,
                                   GParamSpec   *pspec)
{
  FoundryInputSwitch *self = FOUNDRY_INPUT_SWITCH (object);

  switch (prop_id)
    {
    case PROP_VALUE:
      foundry_input_switch_set_value (self, g_value_get_boolean (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_switch_class_init (FoundryInputSwitchClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_input_switch_get_property;
  object_class->set_property = foundry_input_switch_set_property;

  properties[PROP_VALUE] =
    g_param_spec_boolean ("value", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_input_switch_init (FoundryInputSwitch *self)
{
}

gboolean
foundry_input_switch_get_value (FoundryInputSwitch *self)
{
  g_return_val_if_fail (FOUNDRY_IS_INPUT_SWITCH (self), FALSE);

  return self->value;
}

void
foundry_input_switch_set_value (FoundryInputSwitch *self,
                                gboolean            value)
{
  g_return_if_fail (FOUNDRY_IS_INPUT_SWITCH (self));

  value = !!value;

  if (value != self->value)
    {
      self->value = value;
      foundry_notify_pspec_in_main (G_OBJECT (self), properties[PROP_VALUE]);
    }
}

/**
 * foundry_input_switch_new:
 * @validator: (transfer full) (nullable): optional validator
 *
 * Returns: (transfer full):
 */
FoundryInput *
foundry_input_switch_new (const char            *title,
                          const char            *subtitle,
                          FoundryInputValidator *validator,
                          gboolean               value)
{
  g_autoptr(FoundryInputValidator) stolen = NULL;

  g_return_val_if_fail (!validator || FOUNDRY_IS_INPUT_VALIDATOR (validator), NULL);

  stolen = validator;

  return g_object_new (FOUNDRY_TYPE_INPUT_SWITCH,
                       "title", title,
                       "subtitle", subtitle,
                       "validator", validator,
                       "value", value,
                       NULL);
}
