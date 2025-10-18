/* foundry-directory-item.c
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

#include "foundry-directory-item-private.h"
#include "foundry-file-manager.h"

#ifdef FOUNDRY_FEATURE_VCS
# include "foundry-vcs.h"
#endif

enum {
  PROP_0,
  PROP_DIRECTORY,
  PROP_DISPLAY_NAME,
  PROP_FILE,
  PROP_IGNORED,
  PROP_INFO,
  PROP_NAME,
  PROP_SIZE,
  PROP_SYMBOLIC_ICON,
#ifdef FOUNDRY_FEATURE_VCS
  PROP_STATUS,
#endif
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryDirectoryItem, foundry_directory_item, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_directory_item_finalize (GObject *object)
{
  FoundryDirectoryItem *self = (FoundryDirectoryItem *)object;

  g_clear_object (&self->directory);
  g_clear_object (&self->file);
  g_clear_object (&self->info);

  G_OBJECT_CLASS (foundry_directory_item_parent_class)->finalize (object);
}

static void
foundry_directory_item_get_property (GObject    *object,
                                     guint       prop_id,
                                     GValue     *value,
                                     GParamSpec *pspec)
{
  FoundryDirectoryItem *self = FOUNDRY_DIRECTORY_ITEM (object);

  switch (prop_id)
    {
    case PROP_DIRECTORY:
      g_value_take_object (value, foundry_directory_item_dup_directory (self));
      break;

    case PROP_DISPLAY_NAME:
      g_value_take_string (value, foundry_directory_item_dup_display_name (self));
      break;

    case PROP_FILE:
      g_value_take_object (value, foundry_directory_item_dup_file (self));
      break;

    case PROP_IGNORED:
      g_value_set_boolean (value, foundry_directory_item_is_ignored (self));
      break;

    case PROP_INFO:
      g_value_take_object (value, foundry_directory_item_dup_info (self));
      break;

    case PROP_NAME:
      g_value_take_string (value, foundry_directory_item_dup_name (self));
      break;

    case PROP_SIZE:
      g_value_set_uint64 (value, foundry_directory_item_get_size (self));
      break;

    case PROP_SYMBOLIC_ICON:
      g_value_take_object (value, foundry_directory_item_dup_symbolic_icon (self));
      break;

#ifdef FOUNDRY_FEATURE_VCS
    case PROP_STATUS:
      g_value_set_flags (value, foundry_directory_item_get_status (self));
      break;
#endif

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_directory_item_set_property (GObject      *object,
                                     guint         prop_id,
                                     const GValue *value,
                                     GParamSpec   *pspec)
{
  FoundryDirectoryItem *self = FOUNDRY_DIRECTORY_ITEM (object);

  switch (prop_id)
    {
    case PROP_DIRECTORY:
      self->directory = g_value_dup_object (value);
      break;

    case PROP_FILE:
      self->file = g_value_dup_object (value);
      break;

    case PROP_INFO:
      self->info = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_directory_item_class_init (FoundryDirectoryItemClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_directory_item_finalize;
  object_class->get_property = foundry_directory_item_get_property;
  object_class->set_property = foundry_directory_item_set_property;

  properties[PROP_DIRECTORY] =
    g_param_spec_object ("directory", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DISPLAY_NAME] =
    g_param_spec_string ("display-name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_FILE] =
    g_param_spec_object ("file", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_INFO] =
    g_param_spec_object ("info", NULL, NULL,
                         G_TYPE_FILE_INFO,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_IGNORED] =
    g_param_spec_boolean ("ignored", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SIZE] =
    g_param_spec_uint64 ("size", NULL, NULL,
                         0, G_TYPE_UINT64, 0,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SYMBOLIC_ICON] =
    g_param_spec_object ("symbolic-icon", NULL, NULL,
                         G_TYPE_ICON,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

#ifdef FOUNDRY_FEATURE_VCS
  properties[PROP_STATUS] =
    g_param_spec_flags ("status", NULL, NULL,
                        FOUNDRY_TYPE_VCS_FILE_STATUS,
                        FOUNDRY_VCS_FILE_STATUS_CURRENT,
                        (G_PARAM_READABLE |
                         G_PARAM_STATIC_STRINGS));
#endif

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_directory_item_init (FoundryDirectoryItem *self)
{
}

/**
 * foundry_directory_item_dup_file:
 * @self: a [class@Foundry.DirectoryItem]
 *
 * Returns: (transfer full):
 */
GFile *
foundry_directory_item_dup_file (FoundryDirectoryItem *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIRECTORY_ITEM (self), NULL);

  return g_object_ref (self->file);
}

/**
 * foundry_directory_item_dup_directory:
 * @self: a [class@Foundry.DirectoryItem]
 *
 * Returns: (transfer full):
 */
GFile *
foundry_directory_item_dup_directory (FoundryDirectoryItem *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIRECTORY_ITEM (self), NULL);

  return g_object_ref (self->directory);
}

/**
 * foundry_directory_item_dup_info:
 * @self: a [class@Foundry.DirectoryItem]
 *
 * Returns: (transfer full):
 */
GFileInfo *
foundry_directory_item_dup_info (FoundryDirectoryItem *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIRECTORY_ITEM (self), NULL);

  return g_object_ref (self->info);
}

char *
foundry_directory_item_dup_name (FoundryDirectoryItem *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIRECTORY_ITEM (self), NULL);

  if (g_file_info_has_attribute (self->info, G_FILE_ATTRIBUTE_STANDARD_NAME))
    return g_strdup (g_file_info_get_name (self->info));

  return NULL;
}

char *
foundry_directory_item_dup_display_name (FoundryDirectoryItem *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIRECTORY_ITEM (self), NULL);

  if (g_file_info_has_attribute (self->info, G_FILE_ATTRIBUTE_STANDARD_DISPLAY_NAME))
    return g_strdup (g_file_info_get_display_name (self->info));

  return NULL;
}

guint64
foundry_directory_item_get_size (FoundryDirectoryItem *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIRECTORY_ITEM (self), 0);

  if (g_file_info_has_attribute (self->info, G_FILE_ATTRIBUTE_STANDARD_SIZE))
    return g_file_info_get_size (self->info);

  return 0;
}

GFileType
foundry_directory_item_get_file_type (FoundryDirectoryItem *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIRECTORY_ITEM (self), 0);

  if (g_file_info_has_attribute (self->info, G_FILE_ATTRIBUTE_STANDARD_TYPE))
    return g_file_info_get_file_type (self->info);

  return G_FILE_TYPE_REGULAR;
}

char *
foundry_directory_item_dup_content_type (FoundryDirectoryItem *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIRECTORY_ITEM (self), 0);

  if (g_file_info_has_attribute (self->info, G_FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE))
    return g_strdup (g_file_info_get_content_type (self->info));

  return NULL;
}

/**
 * foundry_directory_item_dup_symbolic_icon:
 * @self: a [class@Foundry.DirectoryItem]
 *
 * Returns: (transfer full) (nullable):
 */
GIcon *
foundry_directory_item_dup_symbolic_icon (FoundryDirectoryItem *self)
{
  const char *name = NULL;
  const char *content_type = NULL;

  g_return_val_if_fail (FOUNDRY_IS_DIRECTORY_ITEM (self), 0);

  if (g_file_info_has_attribute (self->info, G_FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE) ||
      g_file_info_has_attribute (self->info, G_FILE_ATTRIBUTE_STANDARD_FAST_CONTENT_TYPE))
    content_type = g_file_info_get_content_type (self->info);

  if (g_file_info_has_attribute (self->info, G_FILE_ATTRIBUTE_STANDARD_NAME))
    name = g_file_info_get_name (self->info);

  if (content_type != NULL && name != NULL)
    return foundry_file_manager_find_symbolic_icon (NULL, content_type, name);

  if (g_file_info_has_attribute (self->info, G_FILE_ATTRIBUTE_STANDARD_SYMBOLIC_ICON))
    {
      GIcon *icon = g_file_info_get_symbolic_icon (self->info);

      if (icon != NULL)
        return g_object_ref (icon);
    }

  return NULL;
}

gboolean
foundry_directory_item_is_ignored (FoundryDirectoryItem *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIRECTORY_ITEM (self), FALSE);

  if (g_file_info_has_attribute (self->info, "vcs::ignored"))
    return g_file_info_get_attribute_boolean (self->info, "vcs::ignored");

  return FALSE;
}

#ifdef FOUNDRY_FEATURE_VCS
FoundryVcsFileStatus
foundry_directory_item_get_status (FoundryDirectoryItem *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIRECTORY_ITEM (self), 0);

  if (g_file_info_has_attribute (self->info, "vcs::status"))
    return g_file_info_get_attribute_uint32 (self->info, "vcs::status");

  return 0;
}
#endif
