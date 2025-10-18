/* foundry-documentation-bundle.c
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

#include "foundry-documentation-bundle.h"
#include "foundry-operation.h"

enum {
  PROP_0,
  PROP_ID,
  PROP_INSTALLED,
  PROP_SUBTITLE,
  PROP_TITLE,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryDocumentationBundle, foundry_documentation_bundle, FOUNDRY_TYPE_CONTEXTUAL)

static GParamSpec *properties[N_PROPS];

static void
foundry_documentation_bundle_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  FoundryDocumentationBundle *self = FOUNDRY_DOCUMENTATION_BUNDLE (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_take_string (value, foundry_documentation_bundle_dup_id (self));
      break;

    case PROP_INSTALLED:
      g_value_set_boolean (value, foundry_documentation_bundle_get_installed (self));
      break;

    case PROP_SUBTITLE:
      g_value_take_string (value, foundry_documentation_bundle_dup_subtitle (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_documentation_bundle_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_documentation_bundle_class_init (FoundryDocumentationBundleClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_documentation_bundle_get_property;

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_INSTALLED] =
    g_param_spec_boolean ("installed", NULL, NULL,
                          FALSE,
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
foundry_documentation_bundle_init (FoundryDocumentationBundle *self)
{
}

gboolean
foundry_documentation_bundle_get_installed (FoundryDocumentationBundle *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_BUNDLE (self), FALSE);

  if (FOUNDRY_DOCUMENTATION_BUNDLE_GET_CLASS (self)->get_installed)
    return FOUNDRY_DOCUMENTATION_BUNDLE_GET_CLASS (self)->get_installed (self);

  return TRUE;
}

/**
 * foundry_documentation_bundle_dup_id:
 * @self: a [class@Foundry.DocumentationBundle]
 *
 * Returns: (transfer full):
 */
char *
foundry_documentation_bundle_dup_id (FoundryDocumentationBundle *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_BUNDLE (self), NULL);

  return FOUNDRY_DOCUMENTATION_BUNDLE_GET_CLASS (self)->dup_id (self);
}

/**
 * foundry_documentation_bundle_dup_title:
 * @self: a [class@Foundry.DocumentationBundle]
 *
 * Returns: (transfer full):
 */
char *
foundry_documentation_bundle_dup_title (FoundryDocumentationBundle *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_BUNDLE (self), NULL);

  return FOUNDRY_DOCUMENTATION_BUNDLE_GET_CLASS (self)->dup_title (self);
}

/**
 * foundry_documentation_bundle_dup_subtitle:
 * @self: a [class@Foundry.DocumentationBundle]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_documentation_bundle_dup_subtitle (FoundryDocumentationBundle *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_BUNDLE (self), NULL);

  if (FOUNDRY_DOCUMENTATION_BUNDLE_GET_CLASS (self)->dup_subtitle)
    return FOUNDRY_DOCUMENTATION_BUNDLE_GET_CLASS (self)->dup_subtitle (self);

  return NULL;
}

/**
 * foundry_documentation_bundle_install:
 * @self: a [class@Foundry.DocumentationBundle]
 * @operation: a [class@Foundry.Operation] to be updated by the provider
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value or rejects with error.
 */
DexFuture *
foundry_documentation_bundle_install (FoundryDocumentationBundle *self,
                                      FoundryOperation           *operation,
                                      DexCancellable             *cancellable)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_BUNDLE (self));
  dex_return_error_if_fail (FOUNDRY_IS_OPERATION (operation));
  dex_return_error_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (FOUNDRY_DOCUMENTATION_BUNDLE_GET_CLASS (self)->install)
    return FOUNDRY_DOCUMENTATION_BUNDLE_GET_CLASS (self)->install (self, operation, cancellable);

  return dex_future_new_true ();
}

/**
 * foundry_documentation_bundle_dup_tags:
 * @self: a [class@Foundry.DocumentationBundle]
 *
 * Gets tags for the documentation which may be useful to show the
 * user to help them make better selections.
 *
 * Returns: (transfer full) (nullable):
 */
char **
foundry_documentation_bundle_dup_tags (FoundryDocumentationBundle *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_BUNDLE (self), NULL);

  if (FOUNDRY_DOCUMENTATION_BUNDLE_GET_CLASS (self)->dup_tags)
    return FOUNDRY_DOCUMENTATION_BUNDLE_GET_CLASS (self)->dup_tags (self);

  return NULL;
}
