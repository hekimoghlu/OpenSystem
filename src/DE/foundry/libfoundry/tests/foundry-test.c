/* foundry-test.c
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

#include "foundry-command.h"
#include "foundry-test.h"

enum {
  PROP_0,
  PROP_ID,
  PROP_COMMAND,
  PROP_TITLE,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryTest, foundry_test, FOUNDRY_TYPE_CONTEXTUAL)

static GParamSpec *properties[N_PROPS];

static void
foundry_test_get_property (GObject    *object,
                           guint       prop_id,
                           GValue     *value,
                           GParamSpec *pspec)
{
  FoundryTest *self = FOUNDRY_TEST (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_take_string (value, foundry_test_dup_id (self));
      break;

    case PROP_COMMAND:
      g_value_take_object (value, foundry_test_dup_command (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_test_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_test_class_init (FoundryTestClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_test_get_property;

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_COMMAND] =
    g_param_spec_object ("command", NULL, NULL,
                         FOUNDRY_TYPE_COMMAND,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_test_init (FoundryTest *self)
{
}

char *
foundry_test_dup_id (FoundryTest *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEST (self), NULL);

  if (FOUNDRY_TEST_GET_CLASS (self)->dup_id)
    return FOUNDRY_TEST_GET_CLASS (self)->dup_id (self);

  return NULL;
}

char *
foundry_test_dup_title (FoundryTest *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEST (self), NULL);

  if (FOUNDRY_TEST_GET_CLASS (self)->dup_title)
    return FOUNDRY_TEST_GET_CLASS (self)->dup_title (self);

  return NULL;
}

/**
 * foundry_test_dup_command:
 * @self: a [class@Foundry.Test]
 *
 * Returns: (transfer full):
 */
FoundryCommand *
foundry_test_dup_command (FoundryTest *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEST (self), NULL);

  if (FOUNDRY_TEST_GET_CLASS (self)->dup_command)
    return FOUNDRY_TEST_GET_CLASS (self)->dup_command (self);

  return NULL;
}
