/* foundry-input-font.c
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

#include "foundry-input-font.h"
#include "foundry-input-validator.h"
#include "foundry-util-private.h"

struct _FoundryInputFont
{
  FoundryInput parent_instance;
  char *value;
  guint monospace : 1;
};

enum {
  PROP_0,
  PROP_MONOSPACE,
  PROP_VALUE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryInputFont, foundry_input_font, FOUNDRY_TYPE_INPUT)

static GParamSpec *properties[N_PROPS];

static void
foundry_input_font_dispose (GObject *object)
{
  FoundryInputFont *self = (FoundryInputFont *)object;

  g_clear_pointer (&self->value, g_free);

  G_OBJECT_CLASS (foundry_input_font_parent_class)->dispose (object);
}

static void
foundry_input_font_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  FoundryInputFont *self = FOUNDRY_INPUT_FONT (object);

  switch (prop_id)
    {
    case PROP_MONOSPACE:
      g_value_set_boolean (value, foundry_input_font_get_monospace (self));
      break;

    case PROP_VALUE:
      g_value_take_string (value, foundry_input_font_dup_value (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_font_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  FoundryInputFont *self = FOUNDRY_INPUT_FONT (object);

  switch (prop_id)
    {
    case PROP_MONOSPACE:
      self->monospace = g_value_get_boolean (value);
      break;

    case PROP_VALUE:
      foundry_input_font_set_value (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_font_class_init (FoundryInputFontClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_input_font_dispose;
  object_class->get_property = foundry_input_font_get_property;
  object_class->set_property = foundry_input_font_set_property;

  properties[PROP_MONOSPACE] =
    g_param_spec_boolean ("monospace", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_CONSTRUCT_ONLY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_VALUE] =
    g_param_spec_string ("value", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_input_font_init (FoundryInputFont *self)
{
}

/**
 * foundry_input_font_new:
 * @title: the title of the input
 * @subtitle: (nullable): optional subtitle
 * @validator: (transfer full) (nullable): optional validator
 * @value: (nullable): optional initial value
 * @monospace: if only monospace fonts should be choosen
 *
 * Returns: (transfer full):
 */
FoundryInput *
foundry_input_font_new (const char            *title,
                        const char            *subtitle,
                        FoundryInputValidator *validator,
                        const char            *value,
                        gboolean               monospace)
{
  g_autoptr(FoundryInputValidator) stolen = NULL;

  g_return_val_if_fail (!validator || FOUNDRY_IS_INPUT_VALIDATOR (validator), NULL);

  stolen = validator;

  return g_object_new (FOUNDRY_TYPE_INPUT_FONT,
                       "title", title,
                       "subtitle", subtitle,
                       "validator", validator,
                       "value", value,
                       "monospace", monospace,
                       NULL);
}

char *
foundry_input_font_dup_value (FoundryInputFont *self)
{
  g_return_val_if_fail (FOUNDRY_IS_INPUT_FONT (self), NULL);

  return g_strdup (self->value);
}

void
foundry_input_font_set_value (FoundryInputFont *self,
                              const char       *value)
{
  g_return_if_fail (FOUNDRY_IS_INPUT_FONT (self));

  if (g_set_str (&self->value, value))
    foundry_notify_pspec_in_main (G_OBJECT (self), properties[PROP_VALUE]);
}

gboolean
foundry_input_font_get_monospace (FoundryInputFont *self)
{
  g_return_val_if_fail (FOUNDRY_IS_INPUT_FONT (self), FALSE);

  return self->monospace;
}
