/* foundry-string-object.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "foundry-string-object-private.h"

struct _FoundryStringObject
{
  GObject parent_instance;
  char *string;
};

G_DEFINE_FINAL_TYPE (FoundryStringObject, foundry_string_object, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_STRING,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_string_object_finalize (GObject *object)
{
  FoundryStringObject *self = (FoundryStringObject *)object;

  g_clear_pointer (&self->string, g_free);

  G_OBJECT_CLASS (foundry_string_object_parent_class)->finalize (object);
}

static void
foundry_string_object_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundryStringObject *self = FOUNDRY_STRING_OBJECT (object);

  switch (prop_id)
    {
    case PROP_STRING:
      g_value_set_string (value, foundry_string_object_get_string (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_string_object_class_init (FoundryStringObjectClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_string_object_finalize;
  object_class->get_property = foundry_string_object_get_property;

  properties[PROP_STRING] =
    g_param_spec_string ("string", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_string_object_init (FoundryStringObject *self)
{
}

FoundryStringObject *
foundry_string_object_new (const char *string)
{
  FoundryStringObject *self;

  self = g_object_new (FOUNDRY_TYPE_STRING_OBJECT, NULL);
  self->string = g_strdup (string);

  return self;
}

const char *
foundry_string_object_get_string (FoundryStringObject *self)
{
  g_return_val_if_fail (FOUNDRY_IS_STRING_OBJECT (self), NULL);

  return self->string;
}
