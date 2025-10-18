/* foundry-vcs-tag.c
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

#include "foundry-vcs-tag.h"
#include "foundry-util.h"

G_DEFINE_ABSTRACT_TYPE (FoundryVcsTag, foundry_vcs_tag, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_ID,
  PROP_IS_LOCAL,
  PROP_TITLE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_vcs_tag_get_property (GObject    *object,
                              guint       prop_id,
                              GValue     *value,
                              GParamSpec *pspec)
{
  FoundryVcsTag *self = FOUNDRY_VCS_TAG (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_take_string (value, foundry_vcs_tag_dup_id (self));
      break;

    case PROP_IS_LOCAL:
      g_value_set_boolean (value, foundry_vcs_tag_is_local (self));
      break;

    case PROP_TITLE:
      g_value_set_string (value, foundry_vcs_tag_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_vcs_tag_class_init (FoundryVcsTagClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_vcs_tag_get_property;

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_IS_LOCAL] =
    g_param_spec_boolean ("is-local", NULL, NULL,
                          TRUE,
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
foundry_vcs_tag_init (FoundryVcsTag *self)
{
}

gboolean
foundry_vcs_tag_is_local (FoundryVcsTag *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_TAG (self), FALSE);

  if (FOUNDRY_VCS_TAG_GET_CLASS (self)->is_local)
    return FOUNDRY_VCS_TAG_GET_CLASS (self)->is_local (self);

  return TRUE;
}

char *
foundry_vcs_tag_dup_id (FoundryVcsTag *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_TAG (self), NULL);

  if (FOUNDRY_VCS_TAG_GET_CLASS (self)->dup_id)
    return FOUNDRY_VCS_TAG_GET_CLASS (self)->dup_id (self);

  return NULL;
}

char *
foundry_vcs_tag_dup_title (FoundryVcsTag *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_TAG (self), NULL);

  if (FOUNDRY_VCS_TAG_GET_CLASS (self)->dup_title)
    return FOUNDRY_VCS_TAG_GET_CLASS (self)->dup_title (self);

  return NULL;
}

/**
 * foundry_vcs_tag_load_target:
 * @self: a [class@Foundry.VcsTag]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.VcsReference].
 */
DexFuture *
foundry_vcs_tag_load_target (FoundryVcsTag *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS_TAG (self));

  if (FOUNDRY_VCS_TAG_GET_CLASS (self)->load_target)
    return FOUNDRY_VCS_TAG_GET_CLASS (self)->load_target (self);

  return foundry_future_new_not_supported ();
}
