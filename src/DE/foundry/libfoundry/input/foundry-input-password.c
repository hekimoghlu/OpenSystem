/* foundry-input-password.c
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

#include "foundry-input-password.h"
#include "foundry-input-validator.h"
#include "foundry-util-private.h"

typedef struct
{
  GMutex  mutex;
  char   *value;
} FoundryInputPasswordPrivate;

enum {
  PROP_0,
  PROP_VALUE,
  N_PROPS
};

G_DEFINE_TYPE_WITH_PRIVATE (FoundryInputPassword, foundry_input_password, FOUNDRY_TYPE_INPUT)

static GParamSpec *properties[N_PROPS];

static void
foundry_input_password_finalize (GObject *object)
{
  FoundryInputPassword *self = (FoundryInputPassword *)object;
  FoundryInputPasswordPrivate *priv = foundry_input_password_get_instance_private (self);

  g_mutex_clear (&priv->mutex);

  g_clear_pointer (&priv->value, g_free);

  G_OBJECT_CLASS (foundry_input_password_parent_class)->finalize (object);
}

static void
foundry_input_password_get_property (GObject    *object,
                                     guint       prop_id,
                                     GValue     *value,
                                     GParamSpec *pspec)
{
  FoundryInputPassword *self = FOUNDRY_INPUT_PASSWORD (object);

  switch (prop_id)
    {
    case PROP_VALUE:
      g_value_take_string (value, foundry_input_password_dup_value (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_password_set_property (GObject      *object,
                                     guint         prop_id,
                                     const GValue *value,
                                     GParamSpec   *pspec)
{
  FoundryInputPassword *self = FOUNDRY_INPUT_PASSWORD (object);

  switch (prop_id)
    {
    case PROP_VALUE:
      foundry_input_password_set_value (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_password_class_init (FoundryInputPasswordClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_input_password_finalize;
  object_class->get_property = foundry_input_password_get_property;
  object_class->set_property = foundry_input_password_set_property;

  properties[PROP_VALUE] =
    g_param_spec_string ("value", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_input_password_init (FoundryInputPassword *self)
{
  FoundryInputPasswordPrivate *priv = foundry_input_password_get_instance_private (self);

  g_mutex_init (&priv->mutex);
}

/**
 * foundry_input_password_new:
 * @validator: (transfer full) (nullable): optional validator
 *
 * Returns: (transfer full):
 */
FoundryInput *
foundry_input_password_new (const char            *title,
                            const char            *subtitle,
                            FoundryInputValidator *validator,
                            const char            *value)
{
  g_autoptr(FoundryInputValidator) stolen = NULL;

  g_return_val_if_fail (!validator || FOUNDRY_IS_INPUT_VALIDATOR (validator), NULL);

  stolen = validator;

  return g_object_new (FOUNDRY_TYPE_INPUT_PASSWORD,
                       "title", title,
                       "subtitle", subtitle,
                       "validator", validator,
                       "value", value,
                       NULL);
}

char *
foundry_input_password_dup_value (FoundryInputPassword *self)
{
  FoundryInputPasswordPrivate *priv = foundry_input_password_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_INPUT_PASSWORD (self), NULL);

  return g_strdup (priv->value);
}

void
foundry_input_password_set_value (FoundryInputPassword *self,
                                  const char           *value)
{
  FoundryInputPasswordPrivate *priv = foundry_input_password_get_instance_private (self);
  gboolean changed;

  g_return_if_fail (FOUNDRY_IS_INPUT_PASSWORD (self));

  g_mutex_lock (&priv->mutex);
  changed = g_set_str (&priv->value, value);
  g_mutex_unlock (&priv->mutex);

  if (changed)
    foundry_notify_pspec_in_main (G_OBJECT (self), properties[PROP_VALUE]);
}
