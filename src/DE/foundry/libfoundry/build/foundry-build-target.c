/* foundry-build-target.c
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

#include "foundry-build-target.h"

G_DEFINE_ABSTRACT_TYPE (FoundryBuildTarget, foundry_build_target, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_ID,
  PROP_TITLE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_build_target_get_property (GObject    *object,
                                   guint       prop_id,
                                   GValue     *value,
                                   GParamSpec *pspec)
{
  FoundryBuildTarget *self = FOUNDRY_BUILD_TARGET (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_take_string (value, foundry_build_target_dup_id (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_build_target_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_build_target_class_init (FoundryBuildTargetClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_build_target_get_property;

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
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
foundry_build_target_init (FoundryBuildTarget *self)
{
}

char *
foundry_build_target_dup_id (FoundryBuildTarget *self)
{
  g_return_val_if_fail (FOUNDRY_IS_BUILD_TARGET (self), NULL);

  if (FOUNDRY_BUILD_TARGET_GET_CLASS (self)->dup_id)
    return FOUNDRY_BUILD_TARGET_GET_CLASS (self)->dup_id (self);

  return NULL;
}

char *
foundry_build_target_dup_title (FoundryBuildTarget *self)
{
  g_return_val_if_fail (FOUNDRY_IS_BUILD_TARGET (self), NULL);

  if (FOUNDRY_BUILD_TARGET_GET_CLASS (self)->dup_title)
    return FOUNDRY_BUILD_TARGET_GET_CLASS (self)->dup_title (self);

  return NULL;
}
