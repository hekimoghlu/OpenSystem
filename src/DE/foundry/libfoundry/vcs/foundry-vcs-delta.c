/* foundry-vcs-delta.c
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

#include "foundry-vcs-delta.h"

G_DEFINE_ABSTRACT_TYPE (FoundryVcsDelta, foundry_vcs_delta, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_OLD_PATH,
  PROP_NEW_PATH,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_vcs_delta_get_property (GObject    *object,
                                guint       prop_id,
                                GValue     *value,
                                GParamSpec *pspec)
{
  FoundryVcsDelta *self = FOUNDRY_VCS_DELTA (object);

  switch (prop_id)
    {
    case PROP_OLD_PATH:
      g_value_take_string (value, foundry_vcs_delta_dup_old_path (self));
      break;

    case PROP_NEW_PATH:
      g_value_take_string (value, foundry_vcs_delta_dup_new_path (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_vcs_delta_class_init (FoundryVcsDeltaClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_vcs_delta_get_property;

  properties[PROP_OLD_PATH] =
    g_param_spec_string ("old-path", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_NEW_PATH] =
    g_param_spec_string ("new-path", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_vcs_delta_init (FoundryVcsDelta *self)
{
}

char *
foundry_vcs_delta_dup_old_path (FoundryVcsDelta *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_DELTA (self), NULL);

  if (FOUNDRY_VCS_DELTA_GET_CLASS (self)->dup_old_path)
    return FOUNDRY_VCS_DELTA_GET_CLASS (self)->dup_old_path (self);

  return NULL;
}

char *
foundry_vcs_delta_dup_new_path (FoundryVcsDelta *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_DELTA (self), NULL);

  if (FOUNDRY_VCS_DELTA_GET_CLASS (self)->dup_new_path)
    return FOUNDRY_VCS_DELTA_GET_CLASS (self)->dup_new_path (self);

  return NULL;
}
