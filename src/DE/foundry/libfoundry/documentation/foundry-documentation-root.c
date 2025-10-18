/* foundry-documentation-root.c
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

#include "foundry-documentation-root.h"

struct _FoundryDocumentationRoot
{
  GObject     parent_instance;
  GIcon      *icon;
  char       *identifier;
  char       *title;
  char       *version;
  GListModel *directories;
};

enum {
  PROP_0,
  PROP_DIRECTORIES,
  PROP_ICON,
  PROP_IDENTIFIER,
  PROP_TITLE,
  PROP_VERSION,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryDocumentationRoot, foundry_documentation_root, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_documentation_root_finalize (GObject *object)
{
  FoundryDocumentationRoot *self = (FoundryDocumentationRoot *)object;

  g_clear_pointer (&self->title, g_free);
  g_clear_pointer (&self->identifier, g_free);
  g_clear_pointer (&self->version, g_free);
  g_clear_object (&self->icon);
  g_clear_object (&self->directories);

  G_OBJECT_CLASS (foundry_documentation_root_parent_class)->finalize (object);
}

static void
foundry_documentation_root_get_property (GObject    *object,
                                         guint       prop_id,
                                         GValue     *value,
                                         GParamSpec *pspec)
{
  FoundryDocumentationRoot *self = FOUNDRY_DOCUMENTATION_ROOT (object);

  switch (prop_id)
    {
    case PROP_DIRECTORIES:
      g_value_take_object (value, foundry_documentation_root_list_directories (self));
      break;

    case PROP_ICON:
      g_value_take_object (value, foundry_documentation_root_dup_icon (self));
      break;

    case PROP_IDENTIFIER:
      g_value_take_string (value, foundry_documentation_root_dup_identifier (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_documentation_root_dup_title (self));
      break;

    case PROP_VERSION:
      g_value_take_string (value, foundry_documentation_root_dup_version (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_documentation_root_set_property (GObject      *object,
                                         guint         prop_id,
                                         const GValue *value,
                                         GParamSpec   *pspec)
{
  FoundryDocumentationRoot *self = FOUNDRY_DOCUMENTATION_ROOT (object);

  switch (prop_id)
    {
    case PROP_DIRECTORIES:
      self->directories = g_value_dup_object (value);
      break;

    case PROP_ICON:
      self->icon = g_value_dup_object (value);
      break;

    case PROP_IDENTIFIER:
      self->identifier = g_value_dup_string (value);
      break;

    case PROP_TITLE:
      self->title = g_value_dup_string (value);
      break;

    case PROP_VERSION:
      self->version = g_value_dup_string (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_documentation_root_class_init (FoundryDocumentationRootClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_documentation_root_finalize;
  object_class->get_property = foundry_documentation_root_get_property;
  object_class->set_property = foundry_documentation_root_set_property;

  properties[PROP_DIRECTORIES] =
    g_param_spec_object ("directories", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ICON] =
    g_param_spec_object ("icon", NULL, NULL,
                         G_TYPE_ICON,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_IDENTIFIER] =
    g_param_spec_string ("identifier", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_VERSION] =
    g_param_spec_string ("version", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_documentation_root_init (FoundryDocumentationRoot *self)
{
}

FoundryDocumentationRoot *
foundry_documentation_root_new (const char *identifier,
                                const char *title,
                                const char *version,
                                GIcon      *icon,
                                GListModel *directories)
{
  g_return_val_if_fail (identifier != NULL, NULL);
  g_return_val_if_fail (title != NULL, NULL);
  g_return_val_if_fail (!icon || G_IS_ICON (icon), NULL);
  g_return_val_if_fail (G_IS_LIST_MODEL (directories), NULL);

  return g_object_new (FOUNDRY_TYPE_DOCUMENTATION_ROOT,
                       "identifier", identifier,
                       "title", title,
                       "version", version,
                       "icon", icon,
                       "directories", directories,
                       NULL);
}

char *
foundry_documentation_root_dup_title (FoundryDocumentationRoot *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_ROOT (self), NULL);

  return g_strdup (self->title);
}

char *
foundry_documentation_root_dup_version (FoundryDocumentationRoot *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_ROOT (self), NULL);

  return g_strdup (self->version);
}

char *
foundry_documentation_root_dup_identifier (FoundryDocumentationRoot *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_ROOT (self), NULL);

  return g_strdup (self->identifier);
}

/**
 * foundry_documentation_root_list_directories:
 * @self: a [class@Foundry.DocumentationRoot]
 *
 * Returns: (transfer full) (not nullable): a [iface@Gio.ListModel] of
 *   [iface@Gio.File].
 */
GListModel *
foundry_documentation_root_list_directories (FoundryDocumentationRoot *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_ROOT (self), NULL);

  return g_object_ref (self->directories);
}

/**
 * foundry_documentation_root_dup_icon:
 * @self: a [class@Foundry.DocumentationRoot]
 *
 * Returns: (transfer full) (nullable):
 */
GIcon *
foundry_documentation_root_dup_icon (FoundryDocumentationRoot *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_ROOT (self), NULL);

  return self->icon ? g_object_ref (self->icon) : NULL;
}
