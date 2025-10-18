/* foundry-tweak.c
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

#include "foundry-context.h"
#include "foundry-tweak.h"

enum {
  PROP_0,
  PROP_DISPLAY_HINT,
  PROP_ICON,
  PROP_PATH,
  PROP_SECTION,
  PROP_SUBTITLE,
  PROP_TITLE,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryTweak, foundry_tweak, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_tweak_get_property (GObject    *object,
                            guint       prop_id,
                            GValue     *value,
                            GParamSpec *pspec)
{
  FoundryTweak *self = FOUNDRY_TWEAK (object);

  switch (prop_id)
    {
    case PROP_DISPLAY_HINT:
      g_value_take_string (value, foundry_tweak_dup_display_hint (self));
      break;

    case PROP_ICON:
      g_value_take_object (value, foundry_tweak_dup_icon (self));
      break;

    case PROP_PATH:
      g_value_take_string (value, foundry_tweak_dup_path (self));
      break;

    case PROP_SECTION:
      g_value_take_string (value, foundry_tweak_dup_section (self));
      break;

    case PROP_SUBTITLE:
      g_value_take_string (value, foundry_tweak_dup_subtitle (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_tweak_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_tweak_class_init (FoundryTweakClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_tweak_get_property;

  properties[PROP_DISPLAY_HINT] =
    g_param_spec_string ("display-hint", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ICON] =
    g_param_spec_object ("icon", NULL, NULL,
                         G_TYPE_ICON,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PATH] =
    g_param_spec_string ("path", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SECTION] =
    g_param_spec_string ("section", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SUBTITLE] =
    g_param_spec_string ("subtitle", NULL, NULL,
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
foundry_tweak_init (FoundryTweak *self)
{
}

char *
foundry_tweak_dup_display_hint (FoundryTweak *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TWEAK (self), NULL);

  if (FOUNDRY_TWEAK_GET_CLASS (self)->dup_display_hint)
    return FOUNDRY_TWEAK_GET_CLASS (self)->dup_display_hint (self);

  return NULL;
}

/**
 * foundry_tweak_dup_icon:
 * @self: a [class@Foundry.Tweak]
 *
 * Returns: (transfer full) (nullable):
 */
GIcon *
foundry_tweak_dup_icon (FoundryTweak *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TWEAK (self), NULL);

  if (FOUNDRY_TWEAK_GET_CLASS (self)->dup_icon)
    return FOUNDRY_TWEAK_GET_CLASS (self)->dup_icon (self);

  return NULL;
}

char *
foundry_tweak_dup_path (FoundryTweak *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TWEAK (self), NULL);

  if (FOUNDRY_TWEAK_GET_CLASS (self)->dup_path)
    return FOUNDRY_TWEAK_GET_CLASS (self)->dup_path (self);

  return NULL;
}

char *
foundry_tweak_dup_title (FoundryTweak *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TWEAK (self), NULL);

  if (FOUNDRY_TWEAK_GET_CLASS (self)->dup_title)
    return FOUNDRY_TWEAK_GET_CLASS (self)->dup_title (self);

  return NULL;
}

char *
foundry_tweak_dup_subtitle (FoundryTweak *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TWEAK (self), NULL);

  if (FOUNDRY_TWEAK_GET_CLASS (self)->dup_subtitle)
    return FOUNDRY_TWEAK_GET_CLASS (self)->dup_subtitle (self);

  return NULL;
}

char *
foundry_tweak_dup_section (FoundryTweak *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TWEAK (self), NULL);

  if (FOUNDRY_TWEAK_GET_CLASS (self)->dup_section)
    return FOUNDRY_TWEAK_GET_CLASS (self)->dup_section (self);

  return NULL;
}

/**
 * foundry_tweak_create_input:
 * @self: a [class@Foundry.Tweak]
 * @context: a [class@Foundry.Context]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryInput *
foundry_tweak_create_input (FoundryTweak   *self,
                            FoundryContext *context)
{
  g_return_val_if_fail (FOUNDRY_IS_TWEAK (self), NULL);
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);

  if (FOUNDRY_TWEAK_GET_CLASS (self)->create_input)
    return FOUNDRY_TWEAK_GET_CLASS (self)->create_input (self, context);

  return NULL;
}

char *
foundry_tweak_dup_sort_key (FoundryTweak *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TWEAK (self), NULL);

  if (FOUNDRY_TWEAK_GET_CLASS (self)->dup_sort_key)
    return FOUNDRY_TWEAK_GET_CLASS (self)->dup_sort_key (self);

  return NULL;
}
