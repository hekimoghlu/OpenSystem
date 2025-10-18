/* foundry-flatpak-manifest.c
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

#include <json-glib/json-glib.h>

#include <foundry.h>

#include "foundry-flatpak-extensions.h"
#include "foundry-flatpak-manifest.h"
#include "foundry-flatpak-modules.h"
#include "foundry-flatpak-options.h"

struct _FoundryFlatpakManifest
{
  FoundryFlatpakSerializable parent_instance;

  char *appdata_license;
  char *base;
  char *base_commit;
  char *base_version;
  char *branch;
  char *collection_id;
  char *command;
  char *default_branch;
  char *desktop_file_name_prefix;
  char *desktop_file_name_suffix;
  char *extension_tag;
  char *id;
  char *id_platform;
  char *metadata;
  char *metadata_platform;
  char *rename_appdata_file;
  char *rename_desktop_file;
  char *rename_icon;
  char *rename_mime_file;
  char *runtime;
  char *runtime_commit;
  char *runtime_version;
  char *sdk;
  char *sdk_commit;
  char *var;

  char **base_extensions;
  char **cleanup;
  char **cleanup_commands;
  char **cleanup_platform;
  char **cleanup_platform_commands;
  char **finish_args;
  char **inherit_extensions;
  char **inherit_sdk_extensions;
  char **platform_extensions;
  char **prepare_platform_commands;
  char **rename_mime_icons;
  char **sdk_extensions;
  char **tags;

  FoundryFlatpakExtensions *add_build_extensions;
  FoundryFlatpakExtensions *add_extensions;
  FoundryFlatpakModules *modules;
  FoundryFlatpakOptions *build_options;

  guint appstream_compose : 1;
  guint build_extension : 1;
  guint build_runtime : 1;
  guint copy_icon : 1;
  guint separate_locales : 1;
  guint writable_sdk : 1;
};

enum {
  PROP_0,
  PROP_ADD_BUILD_EXTENSIONS,
  PROP_ADD_EXTENSIONS,
  PROP_APPDATA_LICENSE,
  PROP_APPSTREAM_COMPOSE,
  PROP_APP_ID,
  PROP_BASE,
  PROP_BASE_COMMIT,
  PROP_BASE_EXTENSIONS,
  PROP_BASE_VERSION,
  PROP_BRANCH,
  PROP_BUILD_EXTENSION,
  PROP_BUILD_OPTIONS,
  PROP_BUILD_RUNTIME,
  PROP_CLEANUP,
  PROP_CLEANUP_COMMANDS,
  PROP_CLEANUP_PLATFORM,
  PROP_CLEANUP_PLATFORM_COMMANDS,
  PROP_COLLECTION_ID,
  PROP_COMMAND,
  PROP_COPY_ICON,
  PROP_DEFAULT_BRANCH,
  PROP_DESKTOP_FILE_NAME_PREFIX,
  PROP_DESKTOP_FILE_NAME_SUFFIX,
  PROP_EXTENSION_TAG,
  PROP_FINISH_ARGS,
  PROP_ID,
  PROP_ID_PLATFORM,
  PROP_INHERIT_EXTENSIONS,
  PROP_INHERIT_SDK_EXTENSIONS,
  PROP_METADATA,
  PROP_METADATA_PLATFORM,
  PROP_MODULES,
  PROP_PLATFORM_EXTENSIONS,
  PROP_PREPARE_PLATFORM_COMMANDS,
  PROP_RENAME_APPDATA_FILE,
  PROP_RENAME_DESKTOP_FILE,
  PROP_RENAME_ICON,
  PROP_RENAME_MIME_FILE,
  PROP_RENAME_MIME_ICONS,
  PROP_RUNTIME,
  PROP_RUNTIME_COMMIT,
  PROP_RUNTIME_VERSION,
  PROP_SDK,
  PROP_SDK_COMMIT,
  PROP_SDK_EXTENSIONS,
  PROP_SEPARATE_LOCALES,
  PROP_TAGS,
  PROP_VAR,
  PROP_WRITABLE_SDK,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakManifest, foundry_flatpak_manifest, FOUNDRY_TYPE_FLATPAK_SERIALIZABLE)

static void
foundry_flatpak_manifest_finalize (GObject *object)
{
  FoundryFlatpakManifest *self = (FoundryFlatpakManifest *)object;

  g_clear_pointer (&self->appdata_license, g_free);
  g_clear_pointer (&self->base, g_free);
  g_clear_pointer (&self->base_commit, g_free);
  g_clear_pointer (&self->base_version, g_free);
  g_clear_pointer (&self->branch, g_free);
  g_clear_pointer (&self->collection_id, g_free);
  g_clear_pointer (&self->command, g_free);
  g_clear_pointer (&self->default_branch, g_free);
  g_clear_pointer (&self->desktop_file_name_prefix, g_free);
  g_clear_pointer (&self->desktop_file_name_suffix, g_free);
  g_clear_pointer (&self->extension_tag, g_free);
  g_clear_pointer (&self->id, g_free);
  g_clear_pointer (&self->id_platform, g_free);
  g_clear_pointer (&self->metadata, g_free);
  g_clear_pointer (&self->metadata_platform, g_free);
  g_clear_pointer (&self->rename_appdata_file, g_free);
  g_clear_pointer (&self->rename_desktop_file, g_free);
  g_clear_pointer (&self->rename_icon, g_free);
  g_clear_pointer (&self->rename_mime_file, g_free);
  g_clear_pointer (&self->runtime, g_free);
  g_clear_pointer (&self->runtime_commit, g_free);
  g_clear_pointer (&self->runtime_version, g_free);
  g_clear_pointer (&self->sdk, g_free);
  g_clear_pointer (&self->sdk_commit, g_free);
  g_clear_pointer (&self->var, g_free);

  g_clear_pointer (&self->base_extensions, g_strfreev);
  g_clear_pointer (&self->cleanup, g_strfreev);
  g_clear_pointer (&self->cleanup_commands, g_strfreev);
  g_clear_pointer (&self->cleanup_platform, g_strfreev);
  g_clear_pointer (&self->cleanup_platform_commands, g_strfreev);
  g_clear_pointer (&self->finish_args, g_strfreev);
  g_clear_pointer (&self->inherit_extensions, g_strfreev);
  g_clear_pointer (&self->inherit_sdk_extensions, g_strfreev);
  g_clear_pointer (&self->platform_extensions, g_strfreev);
  g_clear_pointer (&self->prepare_platform_commands, g_strfreev);
  g_clear_pointer (&self->rename_mime_icons, g_strfreev);
  g_clear_pointer (&self->sdk_extensions, g_strfreev);
  g_clear_pointer (&self->tags, g_strfreev);

  g_clear_object (&self->add_build_extensions);
  g_clear_object (&self->add_extensions);
  g_clear_object (&self->modules);
  g_clear_object (&self->build_options);

  G_OBJECT_CLASS (foundry_flatpak_manifest_parent_class)->finalize (object);
}

static void
foundry_flatpak_manifest_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  FoundryFlatpakManifest *self = FOUNDRY_FLATPAK_MANIFEST (object);

  switch (prop_id)
    {
    case PROP_APP_ID:
      g_value_set_string (value, NULL);
      break;

    case PROP_ID:
      g_value_set_string (value, self->id);
      break;

    case PROP_ID_PLATFORM:
      g_value_set_string (value, self->id_platform);
      break;

    case PROP_BRANCH:
      g_value_set_string (value, self->branch);
      break;

    case PROP_DEFAULT_BRANCH:
      g_value_set_string (value, self->default_branch);
      break;

    case PROP_RUNTIME:
      g_value_set_string (value, self->runtime);
      break;

    case PROP_RUNTIME_COMMIT:
      g_value_set_string (value, self->runtime_commit);
      break;

    case PROP_RUNTIME_VERSION:
      g_value_set_string (value, self->runtime_version);
      break;

    case PROP_SDK:
      g_value_set_string (value, self->sdk);
      break;

    case PROP_SDK_COMMIT:
      g_value_set_string (value, self->sdk_commit);
      break;

    case PROP_BASE:
      g_value_set_string (value, self->base);
      break;

    case PROP_BASE_COMMIT:
      g_value_set_string (value, self->base_commit);
      break;

    case PROP_BASE_VERSION:
      g_value_set_string (value, self->base_version);
      break;

    case PROP_BASE_EXTENSIONS:
      g_value_set_boxed (value, self->base_extensions);
      break;

    case PROP_VAR:
      g_value_set_string (value, self->var);
      break;

    case PROP_METADATA:
      g_value_set_string (value, self->metadata);
      break;

    case PROP_METADATA_PLATFORM:
      g_value_set_string (value, self->metadata_platform);
      break;

    case PROP_COMMAND:
      g_value_set_string (value, self->command);
      break;

    case PROP_BUILD_OPTIONS:
      g_value_set_object (value, self->build_options);
      break;

    case PROP_MODULES:
      g_value_set_object (value, self->modules);
      break;

    case PROP_ADD_EXTENSIONS:
      g_value_set_object (value, self->add_extensions);
      break;

    case PROP_ADD_BUILD_EXTENSIONS:
      g_value_set_object (value, self->add_build_extensions);
      break;

    case PROP_CLEANUP:
      g_value_set_boxed (value, self->cleanup);
      break;

    case PROP_CLEANUP_COMMANDS:
      g_value_set_boxed (value, self->cleanup_commands);
      break;

    case PROP_CLEANUP_PLATFORM:
      g_value_set_boxed (value, self->cleanup_platform);
      break;

    case PROP_CLEANUP_PLATFORM_COMMANDS:
      g_value_set_boxed (value, self->cleanup_platform_commands);
      break;

    case PROP_PREPARE_PLATFORM_COMMANDS:
      g_value_set_boxed (value, self->prepare_platform_commands);
      break;

    case PROP_FINISH_ARGS:
      g_value_set_boxed (value, self->finish_args);
      break;

    case PROP_INHERIT_EXTENSIONS:
      g_value_set_boxed (value, self->inherit_extensions);
      break;

    case PROP_INHERIT_SDK_EXTENSIONS:
      g_value_set_boxed (value, self->inherit_sdk_extensions);
      break;

    case PROP_TAGS:
      g_value_set_boxed (value, self->tags);
      break;

    case PROP_BUILD_RUNTIME:
      g_value_set_boolean (value, self->build_runtime);
      break;

    case PROP_BUILD_EXTENSION:
      g_value_set_boolean (value, self->build_extension);
      break;

    case PROP_SEPARATE_LOCALES:
      g_value_set_boolean (value, self->separate_locales);
      break;

    case PROP_WRITABLE_SDK:
      g_value_set_boolean (value, self->writable_sdk);
      break;

    case PROP_APPSTREAM_COMPOSE:
      g_value_set_boolean (value, self->appstream_compose);
      break;

    case PROP_SDK_EXTENSIONS:
      g_value_set_boxed (value, self->sdk_extensions);
      break;

    case PROP_PLATFORM_EXTENSIONS:
      g_value_set_boxed (value, self->platform_extensions);
      break;

    case PROP_COPY_ICON:
      g_value_set_boolean (value, self->copy_icon);
      break;

    case PROP_RENAME_DESKTOP_FILE:
      g_value_set_string (value, self->rename_desktop_file);
      break;

    case PROP_RENAME_APPDATA_FILE:
      g_value_set_string (value, self->rename_appdata_file);
      break;

    case PROP_RENAME_MIME_FILE:
      g_value_set_string (value, self->rename_mime_file);
      break;

    case PROP_APPDATA_LICENSE:
      g_value_set_string (value, self->appdata_license);
      break;

    case PROP_RENAME_ICON:
      g_value_set_string (value, self->rename_icon);
      break;

    case PROP_RENAME_MIME_ICONS:
      g_value_set_boxed (value, self->rename_mime_icons);
      break;

    case PROP_DESKTOP_FILE_NAME_PREFIX:
      g_value_set_string (value, self->desktop_file_name_prefix);
      break;

    case PROP_DESKTOP_FILE_NAME_SUFFIX:
      g_value_set_string (value, self->desktop_file_name_suffix);
      break;

    case PROP_COLLECTION_ID:
      g_value_set_string (value, self->collection_id);
      break;

    case PROP_EXTENSION_TAG:
      g_value_set_string (value, self->extension_tag);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_manifest_set_property (GObject      *object,
                                      guint         prop_id,
                                      const GValue *value,
                                      GParamSpec   *pspec)
{
  FoundryFlatpakManifest *self = FOUNDRY_FLATPAK_MANIFEST (object);

  switch (prop_id)
    {
    case PROP_APP_ID:
      g_set_str (&self->id, g_value_get_string (value));
      break;

    case PROP_ID:
      g_set_str (&self->id, g_value_get_string (value));
      break;

    case PROP_ID_PLATFORM:
      g_set_str (&self->id_platform, g_value_get_string (value));
      break;

    case PROP_BRANCH:
      g_set_str (&self->branch, g_value_get_string (value));
      break;

    case PROP_DEFAULT_BRANCH:
      g_set_str (&self->default_branch, g_value_get_string (value));
      break;

    case PROP_RUNTIME:
      g_set_str (&self->runtime, g_value_get_string (value));
      break;

    case PROP_RUNTIME_COMMIT:
      g_set_str (&self->runtime_commit, g_value_get_string (value));
      break;

    case PROP_RUNTIME_VERSION:
      g_set_str (&self->runtime_version, g_value_get_string (value));
      break;

    case PROP_SDK:
      g_set_str (&self->sdk, g_value_get_string (value));
      break;

    case PROP_SDK_COMMIT:
      g_set_str (&self->sdk_commit, g_value_get_string (value));
      break;

    case PROP_BASE:
      g_set_str (&self->base, g_value_get_string (value));
      break;

    case PROP_BASE_COMMIT:
      g_set_str (&self->base_commit, g_value_get_string (value));
      break;

    case PROP_BASE_VERSION:
      g_set_str (&self->base_version, g_value_get_string (value));
      break;

    case PROP_BASE_EXTENSIONS:
      foundry_set_strv (&self->base_extensions, g_value_get_boxed (value));
      break;

    case PROP_VAR:
      g_set_str (&self->var, g_value_get_string (value));
      break;

    case PROP_METADATA:
      g_set_str (&self->metadata, g_value_get_string (value));
      break;

    case PROP_METADATA_PLATFORM:
      g_set_str (&self->metadata_platform, g_value_get_string (value));
      break;

    case PROP_COMMAND:
      g_set_str (&self->command, g_value_get_string (value));
      break;

    case PROP_BUILD_OPTIONS:
      g_set_object (&self->build_options,  g_value_get_object (value));
      break;

    case PROP_MODULES:
      g_set_object (&self->modules, g_value_get_object (value));
      break;

    case PROP_ADD_EXTENSIONS:
      g_set_object (&self->add_extensions, g_value_get_object (value));
      break;

    case PROP_ADD_BUILD_EXTENSIONS:
      g_set_object (&self->add_build_extensions, g_value_get_object (value));
      break;

    case PROP_CLEANUP:
      foundry_set_strv (&self->cleanup, g_value_get_boxed (value));
      break;

    case PROP_CLEANUP_COMMANDS:
      foundry_set_strv (&self->cleanup_commands, g_value_get_boxed (value));
      break;

    case PROP_CLEANUP_PLATFORM:
      foundry_set_strv (&self->cleanup_platform, g_value_get_boxed (value));
      break;

    case PROP_CLEANUP_PLATFORM_COMMANDS:
      foundry_set_strv (&self->cleanup_platform_commands, g_value_get_boxed (value));
      break;

    case PROP_PREPARE_PLATFORM_COMMANDS:
      foundry_set_strv (&self->prepare_platform_commands, g_value_get_boxed (value));
      break;

    case PROP_FINISH_ARGS:
      foundry_set_strv (&self->finish_args, g_value_get_boxed (value));
      break;

    case PROP_INHERIT_EXTENSIONS:
      foundry_set_strv (&self->inherit_extensions, g_value_get_boxed (value));
      break;

    case PROP_INHERIT_SDK_EXTENSIONS:
      foundry_set_strv (&self->inherit_sdk_extensions, g_value_get_boxed (value));
      break;

    case PROP_TAGS:
      foundry_set_strv (&self->tags, g_value_get_boxed (value));
      break;

    case PROP_BUILD_RUNTIME:
      self->build_runtime = g_value_get_boolean (value);
      break;

    case PROP_BUILD_EXTENSION:
      self->build_extension = g_value_get_boolean (value);
      break;

    case PROP_SEPARATE_LOCALES:
      self->separate_locales = g_value_get_boolean (value);
      break;

    case PROP_WRITABLE_SDK:
      self->writable_sdk = g_value_get_boolean (value);
      break;

    case PROP_APPSTREAM_COMPOSE:
      self->appstream_compose = g_value_get_boolean (value);
      break;

    case PROP_SDK_EXTENSIONS:
      foundry_set_strv (&self->sdk_extensions, g_value_get_boxed (value));
      break;

    case PROP_PLATFORM_EXTENSIONS:
      foundry_set_strv (&self->platform_extensions, g_value_get_boxed (value));
      break;

    case PROP_COPY_ICON:
      self->copy_icon = g_value_get_boolean (value);
      break;

    case PROP_RENAME_DESKTOP_FILE:
      g_set_str (&self->rename_desktop_file, g_value_get_string (value));
      break;

    case PROP_RENAME_APPDATA_FILE:
      g_set_str (&self->rename_appdata_file, g_value_get_string (value));
      break;

    case PROP_RENAME_MIME_FILE:
      g_set_str (&self->rename_mime_file, g_value_get_string (value));
      break;

    case PROP_APPDATA_LICENSE:
      g_set_str (&self->appdata_license, g_value_get_string (value));
      break;

    case PROP_RENAME_ICON:
      g_set_str (&self->rename_icon, g_value_get_string (value));
      break;

    case PROP_RENAME_MIME_ICONS:
      foundry_set_strv (&self->rename_mime_icons, g_value_get_boxed (value));
      break;

    case PROP_DESKTOP_FILE_NAME_PREFIX:
      g_set_str (&self->desktop_file_name_prefix, g_value_get_string (value));
      break;

    case PROP_DESKTOP_FILE_NAME_SUFFIX:
      g_set_str (&self->desktop_file_name_suffix, g_value_get_string (value));
      break;

    case PROP_COLLECTION_ID:
      g_set_str (&self->collection_id, g_value_get_string (value));
      break;

    case PROP_EXTENSION_TAG:
      g_set_str (&self->extension_tag, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_manifest_class_init (FoundryFlatpakManifestClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_flatpak_manifest_finalize;
  object_class->get_property = foundry_flatpak_manifest_get_property;
  object_class->set_property = foundry_flatpak_manifest_set_property;

  g_object_class_install_property (object_class,
                                   PROP_APP_ID,
                                   g_param_spec_string ("app-id",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_ID,
                                   g_param_spec_string ("id",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_ID_PLATFORM,
                                   g_param_spec_string ("id-platform",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_BRANCH,
                                   g_param_spec_string ("branch",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_DEFAULT_BRANCH,
                                   g_param_spec_string ("default-branch",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_RUNTIME,
                                   g_param_spec_string ("runtime",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_RUNTIME_COMMIT,
                                   g_param_spec_string ("runtime-commit",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_RUNTIME_VERSION,
                                   g_param_spec_string ("runtime-version",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_SDK,
                                   g_param_spec_string ("sdk",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_SDK_COMMIT,
                                   g_param_spec_string ("sdk-commit",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_BASE,
                                   g_param_spec_string ("base",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_BASE_COMMIT,
                                   g_param_spec_string ("base-commit",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_BASE_VERSION,
                                   g_param_spec_string ("base-version",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_BASE_EXTENSIONS,
                                   g_param_spec_boxed ("base-extensions",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_VAR,
                                   g_param_spec_string ("var",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_METADATA,
                                   g_param_spec_string ("metadata",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_METADATA_PLATFORM,
                                   g_param_spec_string ("metadata-platform",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_COMMAND,
                                   g_param_spec_string ("command",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_BUILD_OPTIONS,
                                   g_param_spec_object ("build-options",
                                                        NULL,
                                                        NULL,
                                                        FOUNDRY_TYPE_FLATPAK_OPTIONS,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_MODULES,
                                   g_param_spec_object ("modules",
                                                        NULL,
                                                        NULL,
                                                        FOUNDRY_TYPE_FLATPAK_MODULES,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_ADD_EXTENSIONS,
                                   g_param_spec_object ("add-extensions",
                                                        NULL,
                                                        NULL,
                                                        FOUNDRY_TYPE_FLATPAK_EXTENSIONS,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_ADD_BUILD_EXTENSIONS,
                                   g_param_spec_object ("add-build-extensions",
                                                        NULL,
                                                        NULL,
                                                        FOUNDRY_TYPE_FLATPAK_EXTENSIONS,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_CLEANUP,
                                   g_param_spec_boxed ("cleanup",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_CLEANUP_COMMANDS,
                                   g_param_spec_boxed ("cleanup-commands",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_CLEANUP_PLATFORM,
                                   g_param_spec_boxed ("cleanup-platform",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_CLEANUP_PLATFORM_COMMANDS,
                                   g_param_spec_boxed ("cleanup-platform-commands",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_PREPARE_PLATFORM_COMMANDS,
                                   g_param_spec_boxed ("prepare-platform-commands",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_FINISH_ARGS,
                                   g_param_spec_boxed ("finish-args",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_INHERIT_EXTENSIONS,
                                   g_param_spec_boxed ("inherit-extensions",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_INHERIT_SDK_EXTENSIONS,
                                   g_param_spec_boxed ("inherit-sdk-extensions",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_BUILD_RUNTIME,
                                   g_param_spec_boolean ("build-runtime",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_BUILD_EXTENSION,
                                   g_param_spec_boolean ("build-extension",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_SEPARATE_LOCALES,
                                   g_param_spec_boolean ("separate-locales",
                                                         NULL,
                                                         NULL,
                                                         TRUE,
                                                         G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_WRITABLE_SDK,
                                   g_param_spec_boolean ("writable-sdk",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_APPSTREAM_COMPOSE,
                                   g_param_spec_boolean ("appstream-compose",
                                                         NULL,
                                                         NULL,
                                                         TRUE,
                                                         G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_SDK_EXTENSIONS,
                                   g_param_spec_boxed ("sdk-extensions",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_PLATFORM_EXTENSIONS,
                                   g_param_spec_boxed ("platform-extensions",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_TAGS,
                                   g_param_spec_boxed ("tags",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_RENAME_DESKTOP_FILE,
                                   g_param_spec_string ("rename-desktop-file",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_RENAME_APPDATA_FILE,
                                   g_param_spec_string ("rename-appdata-file",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_RENAME_MIME_FILE,
                                   g_param_spec_string ("rename-mime-file",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_APPDATA_LICENSE,
                                   g_param_spec_string ("appdata-license",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_RENAME_ICON,
                                   g_param_spec_string ("rename-icon",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_RENAME_MIME_ICONS,
                                   g_param_spec_boxed ("rename-mime-icons",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_COPY_ICON,
                                   g_param_spec_boolean ("copy-icon",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_DESKTOP_FILE_NAME_PREFIX,
                                   g_param_spec_string ("desktop-file-name-prefix",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_DESKTOP_FILE_NAME_SUFFIX,
                                   g_param_spec_string ("desktop-file-name-suffix",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_COLLECTION_ID,
                                   g_param_spec_string ("collection-id",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));


  g_object_class_install_property (object_class,
                                   PROP_EXTENSION_TAG,
                                   g_param_spec_string ("extension-tag",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));
}

static void
foundry_flatpak_manifest_init (FoundryFlatpakManifest *self)
{
}

/**
 * foundry_flatpak_manifest_dup_modules:
 * @self: a [class@Foundry.FlatpakManifest]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryFlatpakModules *
foundry_flatpak_manifest_dup_modules (FoundryFlatpakManifest *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST (self), NULL);

  return self->modules ? g_object_ref (self->modules) : NULL;
}

/**
 * foundry_flatpak_manifest_dup_finish_args:
 * @self: a [class@Foundry.FlatpakManifest]
 *
 * Returns: (transfer full) (nullable):
 */
char **
foundry_flatpak_manifest_dup_finish_args (FoundryFlatpakManifest *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST (self), NULL);

  return g_strdupv (self->finish_args);
}

char *
foundry_flatpak_manifest_dup_command (FoundryFlatpakManifest *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST (self), NULL);

  return g_strdup (self->command);
}

/**
 * foundry_flatpak_manifest_dup_build_options:
 * @self: a [class@Foundry.FlatpakManifest]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryFlatpakOptions *
foundry_flatpak_manifest_dup_build_options (FoundryFlatpakManifest *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST (self), NULL);

  return self->build_options ? g_object_ref (self->build_options) : NULL;
}

char *
foundry_flatpak_manifest_dup_id (FoundryFlatpakManifest *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST (self), NULL);

  return g_strdup (self->id);
}

char *
foundry_flatpak_manifest_dup_sdk (FoundryFlatpakManifest *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST (self), NULL);

  return g_strdup (self->sdk);
}

char *
foundry_flatpak_manifest_dup_runtime (FoundryFlatpakManifest *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST (self), NULL);

  return g_strdup (self->runtime);
}

char *
foundry_flatpak_manifest_dup_runtime_version (FoundryFlatpakManifest *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST (self), NULL);

  return g_strdup (self->runtime_version);
}
