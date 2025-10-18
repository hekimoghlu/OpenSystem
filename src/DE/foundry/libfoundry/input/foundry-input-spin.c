/* foundry-input-spin.c
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

#include "foundry-input-spin.h"
#include "foundry-input-validator.h"
#include "foundry-util-private.h"

struct _FoundryInputSpin
{
  FoundryInput parent_instance;
  double value;
  double lower;
  double upper;
  guint n_digits;
};

enum {
  PROP_0,
  PROP_LOWER,
  PROP_N_DIGITS,
  PROP_UPPER,
  PROP_VALUE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryInputSpin, foundry_input_spin, FOUNDRY_TYPE_INPUT)

static GParamSpec *properties[N_PROPS];

static void
foundry_input_spin_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  FoundryInputSpin *self = FOUNDRY_INPUT_SPIN (object);

  switch (prop_id)
    {
    case PROP_N_DIGITS:
      g_value_set_uint (value, foundry_input_spin_get_n_digits (self));
      break;

    case PROP_LOWER:
      g_value_set_double (value, foundry_input_spin_get_lower (self));
      break;

    case PROP_UPPER:
      g_value_set_double (value, foundry_input_spin_get_upper (self));
      break;

    case PROP_VALUE:
      g_value_set_double (value, foundry_input_spin_get_value (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_spin_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  FoundryInputSpin *self = FOUNDRY_INPUT_SPIN (object);

  switch (prop_id)
    {
    case PROP_N_DIGITS:
      self->n_digits = g_value_get_uint (value);
      break;

    case PROP_LOWER:
      self->lower = g_value_get_double (value);
      break;

    case PROP_UPPER:
      self->upper = g_value_get_double (value);
      break;

    case PROP_VALUE:
      foundry_input_spin_set_value (self, g_value_get_double (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_spin_class_init (FoundryInputSpinClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_input_spin_get_property;
  object_class->set_property = foundry_input_spin_set_property;

  properties[PROP_N_DIGITS] =
    g_param_spec_uint ("n-digits", NULL, NULL,
                       0, 12, 0,
                       (G_PARAM_READWRITE |
                        G_PARAM_CONSTRUCT_ONLY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_LOWER] =
    g_param_spec_double ("lower", NULL, NULL,
                         -G_MAXDOUBLE, G_MAXDOUBLE, 0,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_UPPER] =
    g_param_spec_double ("upper", NULL, NULL,
                         -G_MAXDOUBLE, G_MAXDOUBLE, 0,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_VALUE] =
    g_param_spec_double ("value", NULL, NULL,
                         -G_MAXDOUBLE, G_MAXDOUBLE, 0,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_input_spin_init (FoundryInputSpin *self)
{
}

/**
 * foundry_input_spin_new:
 * @title: the title of the input
 * @subtitle: (nullable): optional subtitle
 * @validator: (transfer full) (nullable): optional validator
 *
 * Returns: (transfer full):
 */
FoundryInput *
foundry_input_spin_new (const char            *title,
                        const char            *subtitle,
                        FoundryInputValidator *validator,
                        double                 value,
                        double                 lower,
                        double                 upper,
                        guint                  n_digits)
{
  g_autoptr(FoundryInputValidator) stolen = NULL;

  g_return_val_if_fail (!validator || FOUNDRY_IS_INPUT_VALIDATOR (validator), NULL);

  stolen = validator;

  return g_object_new (FOUNDRY_TYPE_INPUT_SPIN,
                       "title", title,
                       "subtitle", subtitle,
                       "validator", validator,
                       "value", value,
                       "lower", lower,
                       "upper", upper,
                       "n-digits", n_digits,
                       NULL);
}

double
foundry_input_spin_get_value (FoundryInputSpin *self)
{
  g_return_val_if_fail (FOUNDRY_IS_INPUT_SPIN (self), .0);

  return self->value;
}

void
foundry_input_spin_set_value (FoundryInputSpin *self,
                              double            value)
{
  g_return_if_fail (FOUNDRY_IS_INPUT_SPIN (self));

  if (self->value != value)
    {
      self->value = value;
      foundry_notify_pspec_in_main (G_OBJECT (self), properties[PROP_VALUE]);
    }
}

guint
foundry_input_spin_get_n_digits (FoundryInputSpin *self)
{
  g_return_val_if_fail (FOUNDRY_IS_INPUT_SPIN (self), 0);

  return self->n_digits;
}

double
foundry_input_spin_get_lower (FoundryInputSpin *self)
{
  g_return_val_if_fail (FOUNDRY_IS_INPUT_SPIN (self), .0);

  return self->lower;
}

double
foundry_input_spin_get_upper (FoundryInputSpin *self)
{
  g_return_val_if_fail (FOUNDRY_IS_INPUT_SPIN (self), .0);

  return self->upper;
}
