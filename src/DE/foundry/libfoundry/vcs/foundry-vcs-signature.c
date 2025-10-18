/* foundry-vcs-signature.c
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

#include "foundry-vcs-signature.h"

enum {
  PROP_0,
  PROP_EMAIL,
  PROP_NAME,
  PROP_WHEN,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryVcsSignature, foundry_vcs_signature, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_vcs_signature_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundryVcsSignature *self = FOUNDRY_VCS_SIGNATURE (object);

  switch (prop_id)
    {
    case PROP_EMAIL:
      g_value_take_string (value, foundry_vcs_signature_dup_email (self));
      break;

    case PROP_NAME:
      g_value_take_string (value, foundry_vcs_signature_dup_name (self));
      break;

    case PROP_WHEN:
      g_value_take_boxed (value, foundry_vcs_signature_dup_when (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_vcs_signature_class_init (FoundryVcsSignatureClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_vcs_signature_get_property;

  properties[PROP_EMAIL] =
    g_param_spec_string ("email", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_WHEN] =
    g_param_spec_boxed ("when", NULL, NULL,
                        G_TYPE_DATE_TIME,
                        (G_PARAM_READABLE |
                         G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_vcs_signature_init (FoundryVcsSignature *self)
{
}

/**
 * foundry_vcs_signature_dup_email:
 * @self: a [class@Foundry.VcsSignature]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_vcs_signature_dup_email (FoundryVcsSignature *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_SIGNATURE (self), NULL);

  if (FOUNDRY_VCS_SIGNATURE_GET_CLASS (self)->dup_email)
    return FOUNDRY_VCS_SIGNATURE_GET_CLASS (self)->dup_email (self);

  return NULL;
}

/**
 * foundry_vcs_signature_dup_name:
 * @self: a [class@Foundry.VcsSignature]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_vcs_signature_dup_name (FoundryVcsSignature *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_SIGNATURE (self), NULL);

  if (FOUNDRY_VCS_SIGNATURE_GET_CLASS (self)->dup_name)
    return FOUNDRY_VCS_SIGNATURE_GET_CLASS (self)->dup_name (self);

  return NULL;
}

/**
 * foundry_vcs_signature_dup_when:
 * @self: a [class@Foundry.VcsSignature]
 *
 * Returns: (transfer full) (nullable):
 */
GDateTime *
foundry_vcs_signature_dup_when (FoundryVcsSignature *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_SIGNATURE (self), NULL);

  if (FOUNDRY_VCS_SIGNATURE_GET_CLASS (self)->dup_when)
    return FOUNDRY_VCS_SIGNATURE_GET_CLASS (self)->dup_when (self);

  return NULL;
}
