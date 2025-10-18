/* foundry-input.c
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

typedef struct
{
  char                  *subtitle;
  char                  *title;
  FoundryInputValidator *validator;
} FoundryInputPrivate;

enum {
  PROP_0,
  PROP_SUBTITLE,
  PROP_TITLE,
  PROP_VALIDATOR,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryInput, foundry_input, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_input_finalize (GObject *object)
{
  FoundryInput *self = (FoundryInput *)object;
  FoundryInputPrivate *priv = foundry_input_get_instance_private (self);

  g_clear_pointer (&priv->title, g_free);
  g_clear_pointer (&priv->subtitle, g_free);
  g_clear_object (&priv->validator);

  G_OBJECT_CLASS (foundry_input_parent_class)->finalize (object);
}

static void
foundry_input_get_property (GObject    *object,
                            guint       prop_id,
                            GValue     *value,
                            GParamSpec *pspec)
{
  FoundryInput *self = FOUNDRY_INPUT (object);

  switch (prop_id)
    {
    case PROP_SUBTITLE:
      g_value_take_string (value, foundry_input_dup_subtitle (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_input_dup_title (self));
      break;

    case PROP_VALIDATOR:
      g_value_take_object (value, foundry_input_dup_validator (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_set_property (GObject      *object,
                            guint         prop_id,
                            const GValue *value,
                            GParamSpec   *pspec)
{
  FoundryInput *self = FOUNDRY_INPUT (object);
  FoundryInputPrivate *priv = foundry_input_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_SUBTITLE:
      priv->subtitle = g_value_dup_string (value);
      break;

    case PROP_TITLE:
      priv->title = g_value_dup_string (value);
      break;

    case PROP_VALIDATOR:
      priv->validator = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_class_init (FoundryInputClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_input_finalize;
  object_class->get_property = foundry_input_get_property;
  object_class->set_property = foundry_input_set_property;

  properties[PROP_SUBTITLE] =
    g_param_spec_string ("subtitle", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_VALIDATOR] =
    g_param_spec_object ("validator", NULL, NULL,
                         FOUNDRY_TYPE_INPUT_VALIDATOR,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_input_init (FoundryInput *self)
{
}

char *
foundry_input_dup_subtitle (FoundryInput *self)
{
  FoundryInputPrivate *priv = foundry_input_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_INPUT (self), NULL);

  return g_strdup (priv->subtitle);
}

char *
foundry_input_dup_title (FoundryInput *self)
{
  FoundryInputPrivate *priv = foundry_input_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_INPUT (self), NULL);

  return g_strdup (priv->title);
}

/**
 * foundry_input_validate:
 * @self: a [class@Foundry.Input]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value or rejects with error.
 */
DexFuture *
foundry_input_validate (FoundryInput *self)
{
  FoundryInputPrivate *priv = foundry_input_get_instance_private (self);

  dex_return_error_if_fail (FOUNDRY_IS_INPUT (self));

  if (priv->validator != NULL)
    return foundry_input_validator_validate (priv->validator, self);

  return dex_future_new_true ();
}

/**
 * foundry_input_dup_validator:
 * @self: a [class@Foundry.Input]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryInputValidator *
foundry_input_dup_validator (FoundryInput *self)
{
  FoundryInputPrivate *priv = foundry_input_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_INPUT (self), NULL);

  return priv->validator ? g_object_ref (priv->validator) : NULL;
}
