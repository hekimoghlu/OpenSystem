/* foundry-documentation.c
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

#include "foundry-documentation.h"

G_DEFINE_ABSTRACT_TYPE (FoundryDocumentation, foundry_documentation, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_DEPRECATED_IN,
  PROP_ICON,
  PROP_MENU_ICON,
  PROP_MENU_TITLE,
  PROP_SECTION_TITLE,
  PROP_SINCE_VERSION,
  PROP_TITLE,
  PROP_URI,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_documentation_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundryDocumentation *self = FOUNDRY_DOCUMENTATION (object);

  switch (prop_id)
    {
    case PROP_DEPRECATED_IN:
      g_value_take_string (value, foundry_documentation_dup_deprecated_in (self));
      break;

    case PROP_ICON:
      g_value_take_object (value, foundry_documentation_dup_icon (self));
      break;

    case PROP_MENU_ICON:
      g_value_take_object (value, foundry_documentation_dup_menu_icon (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_documentation_dup_title (self));
      break;

    case PROP_SECTION_TITLE:
      g_value_take_string (value, foundry_documentation_dup_section_title (self));
      break;

    case PROP_SINCE_VERSION:
      g_value_take_string (value, foundry_documentation_dup_since_version (self));
      break;

    case PROP_MENU_TITLE:
      g_value_take_string (value, foundry_documentation_dup_menu_title (self));
      break;

    case PROP_URI:
      g_value_take_string (value, foundry_documentation_dup_uri (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_documentation_class_init (FoundryDocumentationClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_documentation_get_property;

  properties[PROP_DEPRECATED_IN] =
    g_param_spec_string ("deprecated-in", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ICON] =
    g_param_spec_object ("icon", NULL, NULL,
                         G_TYPE_ICON,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_MENU_ICON] =
    g_param_spec_object ("menu-icon", NULL, NULL,
                         G_TYPE_ICON,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SECTION_TITLE] =
    g_param_spec_string ("section-title", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_MENU_TITLE] =
    g_param_spec_string ("menu-title", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SINCE_VERSION] =
    g_param_spec_string ("since-version", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_URI] =
    g_param_spec_string ("uri", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_documentation_init (FoundryDocumentation *self)
{
}

/**
 * foundry_documentation_dup_uri:
 * @self: a [class@Foundry.Documentation]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_documentation_dup_uri (FoundryDocumentation *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION (self), NULL);

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_uri)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_uri (self);

  return NULL;
}

/**
 * foundry_documentation_dup_title:
 * @self: a [class@Foundry.Documentation]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_documentation_dup_title (FoundryDocumentation *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION (self), NULL);

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_title)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_title (self);

  return NULL;
}

/**
 * foundry_documentation_dup_section_title:
 * @self: a [class@Foundry.Documentation]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_documentation_dup_section_title (FoundryDocumentation *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION (self), NULL);

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_section_title)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_section_title (self);

  return NULL;
}

/**
 * foundry_documentation_dup_menu_title:
 * @self: a [class@Foundry.Documentation]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_documentation_dup_menu_title (FoundryDocumentation *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION (self), NULL);

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_menu_title)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_menu_title (self);

  return foundry_documentation_dup_title (self);
}

/**
 * foundry_documentation_dup_deprecated_in:
 * @self: a [class@Foundry.Documentation]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_documentation_dup_deprecated_in (FoundryDocumentation *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION (self), NULL);

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_deprecated_in)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_deprecated_in (self);

  return NULL;
}

/**
 * foundry_documentation_dup_since_version:
 * @self: a [class@Foundry.Documentation]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_documentation_dup_since_version (FoundryDocumentation *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION (self), NULL);

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_since_version)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_since_version (self);

  return NULL;
}

/**
 * foundry_documentation_find_parent:
 * @self: a [class@Foundry.Documentation]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [class@Foundry.Documentation] or rejects with error.
 */
DexFuture *
foundry_documentation_find_parent (FoundryDocumentation *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION (self));

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->find_parent)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->find_parent (self);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}

/**
 * foundry_documentation_find_children:
 * @self: a [class@Foundry.Documentation]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] of [class@Foundry.Documentation] or
 *   rejects with error.
 */
DexFuture *
foundry_documentation_find_children (FoundryDocumentation *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION (self));

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->find_children)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->find_children (self);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}

/**
 * foundry_documentation_find_siblings:
 * @self: a [class@Foundry.Documentation]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] of [class@Foundry.Documentation] or
 *   rejects with error.
 */
DexFuture *
foundry_documentation_find_siblings (FoundryDocumentation *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION (self));

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->find_siblings)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->find_siblings (self);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}

/**
 * foundry_documentation_dup_icon:
 * @self: a [class@Foundry.Documentation]
 *
 * Returns: (transfer full) (nullable):
 */
GIcon *
foundry_documentation_dup_icon (FoundryDocumentation *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION (self), NULL);

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_icon)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_icon (self);

  return NULL;
}

/**
 * foundry_documentation_dup_menu_icon:
 * @self: a [class@Foundry.Documentation]
 *
 * Returns: (transfer full) (nullable):
 */
GIcon *
foundry_documentation_dup_menu_icon (FoundryDocumentation *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION (self), NULL);

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_menu_icon)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->dup_menu_icon (self);

  return foundry_documentation_dup_icon (self);
}

gboolean
foundry_documentation_has_children (FoundryDocumentation *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION (self), FALSE);

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->has_children)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->has_children (self);

  return FALSE;
}

/**
 * foundry_documentation_query_attribute:
 * @self: a [class@Foundry.Documentation]
 *
 * Query various attributes which may be part of the documentation
 * but are not required by plugins to implement.
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_documentation_query_attribute (FoundryDocumentation *self,
                                       const char           *attribute)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION (self), NULL);
  g_return_val_if_fail (attribute != NULL, NULL);

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->query_attribute)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->query_attribute (self, attribute);

  return NULL;
}

gboolean
foundry_documentation_equal (FoundryDocumentation *self,
                             FoundryDocumentation *other)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION (self), FALSE);
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION (other), FALSE);

  if (self == other)
    return TRUE;

  if (G_OBJECT_TYPE (self) != G_OBJECT_TYPE (other))
    return FALSE;

  if (FOUNDRY_DOCUMENTATION_GET_CLASS (self)->equal)
    return FOUNDRY_DOCUMENTATION_GET_CLASS (self)->equal (self, other);

  return FALSE;
}
