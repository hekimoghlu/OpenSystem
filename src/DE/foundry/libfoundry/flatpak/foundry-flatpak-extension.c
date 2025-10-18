/* foundry-flatpak-extension.c
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

#include "foundry-flatpak-extension.h"

struct _FoundryFlatpakExtension
{
  FoundryFlatpakSerializable  parent_instance;

  char                       *add_ld_path;
  char                       *autoprune_unless;
  char                       *directory;
  char                       *download_if;
  char                       *enable_if;
  char                       *merge_dirs;
  char                       *name;
  char                       *subdirectory_suffix;
  char                       *version;
  char                       *versions;

  guint                       autodelete : 1;
  guint                       bundle : 1;
  guint                       locale_subset : 1;
  guint                       no_autodownload : 1;
  guint                       remove_after_build : 1;
  guint                       subdirectories : 1;
};

enum {
  PROP_0,
  PROP_ADD_LD_PATH,
  PROP_AUTODELETE,
  PROP_AUTOPRUNE_UNLESS,
  PROP_BUNDLE,
  PROP_DIRECTORY,
  PROP_DOWNLOAD_IF,
  PROP_ENABLE_IF,
  PROP_LOCALE_SUBSET,
  PROP_MERGE_DIRS,
  PROP_NAME,
  PROP_NO_AUTODOWNLOAD,
  PROP_REMOVE_AFTER_BUILD,
  PROP_SUBDIRECTORIES,
  PROP_SUBDIRECTORY_SUFFIX,
  PROP_VERSION,
  PROP_VERSIONS,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakExtension, foundry_flatpak_extension, FOUNDRY_TYPE_FLATPAK_SERIALIZABLE)

static void
foundry_flatpak_extension_finalize (GObject *object)
{
  FoundryFlatpakExtension *self = (FoundryFlatpakExtension *)object;

  g_clear_pointer (&self->name, g_free);
  g_clear_pointer (&self->directory, g_free);
  g_clear_pointer (&self->add_ld_path, g_free);
  g_clear_pointer (&self->download_if, g_free);
  g_clear_pointer (&self->enable_if, g_free);
  g_clear_pointer (&self->autoprune_unless, g_free);
  g_clear_pointer (&self->merge_dirs, g_free);
  g_clear_pointer (&self->subdirectory_suffix, g_free);
  g_clear_pointer (&self->version, g_free);
  g_clear_pointer (&self->versions, g_free);

  G_OBJECT_CLASS (foundry_flatpak_extension_parent_class)->finalize (object);
}

static void
foundry_flatpak_extension_get_property (GObject    *object,
                                        guint       prop_id,
                                        GValue     *value,
                                        GParamSpec *pspec)
{
  FoundryFlatpakExtension *self = FOUNDRY_FLATPAK_EXTENSION (object);

  switch (prop_id)
    {
    case PROP_NAME:
      g_value_set_string (value, self->name);
      break;

    case PROP_DIRECTORY:
      g_value_set_string (value, self->directory);
      break;

    case PROP_BUNDLE:
      g_value_set_boolean (value, self->bundle);
      break;

    case PROP_REMOVE_AFTER_BUILD:
      g_value_set_boolean (value, self->remove_after_build);
      break;

    case PROP_AUTODELETE:
      g_value_set_boolean (value, self->autodelete);
      break;

    case PROP_NO_AUTODOWNLOAD:
      g_value_set_boolean (value, self->no_autodownload);
      break;

    case PROP_LOCALE_SUBSET:
      g_value_set_boolean (value, self->locale_subset);
      break;

    case PROP_SUBDIRECTORIES:
      g_value_set_boolean (value, self->autodelete);
      break;

    case PROP_ADD_LD_PATH:
      g_value_set_string (value, self->add_ld_path);
      break;

    case PROP_DOWNLOAD_IF:
      g_value_set_string (value, self->download_if);
      break;

    case PROP_ENABLE_IF:
      g_value_set_string (value, self->enable_if);
      break;

    case PROP_AUTOPRUNE_UNLESS:
      g_value_set_string (value, self->autoprune_unless);
      break;

    case PROP_MERGE_DIRS:
      g_value_set_string (value, self->merge_dirs);
      break;

    case PROP_SUBDIRECTORY_SUFFIX:
      g_value_set_string (value, self->subdirectory_suffix);
      break;

    case PROP_VERSION:
      g_value_set_string (value, self->version);
      break;

    case PROP_VERSIONS:
      g_value_set_string (value, self->versions);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_extension_set_property (GObject      *object,
                                        guint         prop_id,
                                        const GValue *value,
                                        GParamSpec   *pspec)
{
  FoundryFlatpakExtension *self = FOUNDRY_FLATPAK_EXTENSION (object);

  switch (prop_id)
    {
    case PROP_NAME:
      g_set_str (&self->name, g_value_get_string (value));
      break;

    case PROP_DIRECTORY:
      g_set_str (&self->directory, g_value_get_string (value));
      break;

    case PROP_BUNDLE:
      self->bundle = g_value_get_boolean (value);
      break;

    case PROP_REMOVE_AFTER_BUILD:
      self->remove_after_build = g_value_get_boolean (value);
      break;

    case PROP_AUTODELETE:
      self->autodelete = g_value_get_boolean (value);
      break;

    case PROP_NO_AUTODOWNLOAD:
      self->no_autodownload = g_value_get_boolean (value);
      break;

    case PROP_LOCALE_SUBSET:
      self->locale_subset = g_value_get_boolean (value);
      break;

    case PROP_SUBDIRECTORIES:
      self->subdirectories = g_value_get_boolean (value);
      break;

    case PROP_ADD_LD_PATH:
      g_set_str (&self->add_ld_path, g_value_get_string (value));
      break;

    case PROP_DOWNLOAD_IF:
      g_set_str (&self->download_if, g_value_get_string (value));
      break;

    case PROP_ENABLE_IF:
      g_set_str (&self->enable_if, g_value_get_string (value));
      break;

  case PROP_AUTOPRUNE_UNLESS:
      g_set_str (&self->autoprune_unless, g_value_get_string (value));
      break;

    case PROP_MERGE_DIRS:
      g_set_str (&self->merge_dirs, g_value_get_string (value));
      break;

    case PROP_SUBDIRECTORY_SUFFIX:
      g_set_str (&self->subdirectory_suffix, g_value_get_string (value));
      break;

    case PROP_VERSION:
      g_set_str (&self->version, g_value_get_string (value));
      break;

    case PROP_VERSIONS:
      g_set_str (&self->versions, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_extension_class_init (FoundryFlatpakExtensionClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_flatpak_extension_finalize;
  object_class->get_property = foundry_flatpak_extension_get_property;
  object_class->set_property = foundry_flatpak_extension_set_property;

  g_object_class_install_property (object_class,
                                   PROP_NAME,
                                   g_param_spec_string ("name",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_DIRECTORY,
                                   g_param_spec_string ("directory",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_BUNDLE,
                                   g_param_spec_boolean ("bundle",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_REMOVE_AFTER_BUILD,
                                   g_param_spec_boolean ("remove-after-build",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_AUTODELETE,
                                   g_param_spec_boolean ("autodelete",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_NO_AUTODOWNLOAD,
                                   g_param_spec_boolean ("no-autodownload",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_LOCALE_SUBSET,
                                   g_param_spec_boolean ("locale-subset",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_SUBDIRECTORIES,
                                   g_param_spec_boolean ("subdirectories",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_ADD_LD_PATH,
                                   g_param_spec_string ("add-ld-path",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_DOWNLOAD_IF,
                                   g_param_spec_string ("download-if",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_ENABLE_IF,
                                   g_param_spec_string ("enable-if",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_AUTOPRUNE_UNLESS,
                                   g_param_spec_string ("autoprune-unless",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_MERGE_DIRS,
                                   g_param_spec_string ("merge-dirs",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_SUBDIRECTORY_SUFFIX,
                                   g_param_spec_string ("subdirectory-suffix",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_VERSION,
                                   g_param_spec_string ("version",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_VERSIONS,
                                   g_param_spec_string ("versions",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

static void
foundry_flatpak_extension_init (FoundryFlatpakExtension *self)
{
}
