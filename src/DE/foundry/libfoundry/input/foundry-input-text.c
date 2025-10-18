/* foundry-input-text.c
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

#include "foundry-input-text.h"
#include "foundry-input-validator.h"
#include "foundry-util-private.h"

struct _FoundryInputText
{
  FoundryInput parent_instance;
  char *value;
};

enum {
  PROP_0,
  PROP_VALUE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryInputText, foundry_input_text, FOUNDRY_TYPE_INPUT)

static GParamSpec *properties[N_PROPS];

static void
foundry_input_text_dispose (GObject *object)
{
  FoundryInputText *self = (FoundryInputText *)object;

  g_clear_pointer (&self->value, g_free);

  G_OBJECT_CLASS (foundry_input_text_parent_class)->dispose (object);
}

static void
foundry_input_text_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  FoundryInputText *self = FOUNDRY_INPUT_TEXT (object);

  switch (prop_id)
    {
    case PROP_VALUE:
      g_value_take_string (value, foundry_input_text_dup_value (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_text_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  FoundryInputText *self = FOUNDRY_INPUT_TEXT (object);

  switch (prop_id)
    {
    case PROP_VALUE:
      foundry_input_text_set_value (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_text_class_init (FoundryInputTextClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_input_text_dispose;
  object_class->get_property = foundry_input_text_get_property;
  object_class->set_property = foundry_input_text_set_property;

  properties[PROP_VALUE] =
    g_param_spec_string ("value", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_input_text_init (FoundryInputText *self)
{
}

/**
 * foundry_input_text_new:
 * @title: the title of the input
 * @subtitle: (nullable): optional subtitle
 * @validator: (transfer full) (nullable): optional validator
 * @value: (nullable): optional initial value
 *
 * Returns: (transfer full):
 */
FoundryInput *
foundry_input_text_new (const char            *title,
                        const char            *subtitle,
                        FoundryInputValidator *validator,
                        const char            *value)
{
  g_autoptr(FoundryInputValidator) stolen = NULL;

  g_return_val_if_fail (!validator || FOUNDRY_IS_INPUT_VALIDATOR (validator), NULL);

  stolen = validator;

  return g_object_new (FOUNDRY_TYPE_INPUT_TEXT,
                       "title", title,
                       "subtitle", subtitle,
                       "validator", validator,
                       "value", value,
                       NULL);
}

char *
foundry_input_text_dup_value (FoundryInputText *self)
{
  g_return_val_if_fail (FOUNDRY_IS_INPUT_TEXT (self), NULL);

  return g_strdup (self->value);
}

void
foundry_input_text_set_value (FoundryInputText *self,
                              const char       *value)
{
  g_return_if_fail (FOUNDRY_IS_INPUT_TEXT (self));

  if (g_set_str (&self->value, value))
    foundry_notify_pspec_in_main (G_OBJECT (self), properties[PROP_VALUE]);
}
