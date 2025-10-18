/* foundry-vcs-reference.c
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

#include "foundry-vcs-reference.h"
#include "foundry-util.h"

enum {
  PROP_0,
  PROP_ID,
  PROP_IS_SYMBOLIC,
  PROP_TITLE,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryVcsReference, foundry_vcs_reference, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_vcs_reference_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundryVcsReference *self = FOUNDRY_VCS_REFERENCE (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_take_string (value, foundry_vcs_reference_dup_id (self));
      break;

    case PROP_IS_SYMBOLIC:
      g_value_set_boolean (value, foundry_vcs_reference_is_symbolic (self));
      break;

    case PROP_TITLE:
      g_value_set_string (value, foundry_vcs_reference_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_vcs_reference_class_init (FoundryVcsReferenceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_vcs_reference_get_property;

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

  properties[PROP_IS_SYMBOLIC] =
    g_param_spec_boolean ("is-symbolic", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_vcs_reference_init (FoundryVcsReference *self)
{
}

char *
foundry_vcs_reference_dup_id (FoundryVcsReference *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_REFERENCE (self), NULL);

  if (FOUNDRY_VCS_REFERENCE_GET_CLASS (self)->dup_id)
    return FOUNDRY_VCS_REFERENCE_GET_CLASS (self)->dup_id (self);

  return NULL;
}

char *
foundry_vcs_reference_dup_title (FoundryVcsReference *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_REFERENCE (self), NULL);

  if (FOUNDRY_VCS_REFERENCE_GET_CLASS (self)->dup_title)
    return FOUNDRY_VCS_REFERENCE_GET_CLASS (self)->dup_title (self);

  return NULL;
}

gboolean
foundry_vcs_reference_is_symbolic (FoundryVcsReference *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_REFERENCE (self), FALSE);

  if (FOUNDRY_VCS_REFERENCE_GET_CLASS (self)->is_symbolic)
    return FOUNDRY_VCS_REFERENCE_GET_CLASS (self)->is_symbolic (self);

  return FALSE;
}

/**
 * foundry_vcs_reference_resolve:
 * @self: a [class@Foundry.VcsReference]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.VcsReference] or rejects with error.
 */
DexFuture *
foundry_vcs_reference_resolve (FoundryVcsReference *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS_REFERENCE (self));

  if (FOUNDRY_VCS_REFERENCE_GET_CLASS (self)->resolve)
    return FOUNDRY_VCS_REFERENCE_GET_CLASS (self)->resolve (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_reference_load_commit:
 * @self: a [class@Foundry.VcsReference]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.VcsCommit] or rejects with error.
 */
DexFuture *
foundry_vcs_reference_load_commit (FoundryVcsReference *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS_REFERENCE (self));

  if (FOUNDRY_VCS_REFERENCE_GET_CLASS (self)->load_commit)
    return FOUNDRY_VCS_REFERENCE_GET_CLASS (self)->load_commit (self);

  return foundry_future_new_not_supported ();
}
