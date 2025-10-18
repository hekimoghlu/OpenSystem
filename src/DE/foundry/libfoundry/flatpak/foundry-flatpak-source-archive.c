/* foundry-flatpak-source-archive.c
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

#include "foundry-flatpak-source-archive.h"
#include "foundry-flatpak-source-private.h"
#include "foundry-util.h"

struct _FoundryFlatpakSourceArchive
{
  FoundryFlatpakSource   parent_instance;
  char                  *path;
  char                  *url;
  char                 **mirror_urls;
  char                  *md5;
  char                  *sha1;
  char                  *sha256;
  char                  *sha512;
  char                  *dest_filename;
  char                  *archive_type;
  char                  *http_referer;
  guint                  strip_components;
  guint                  git_init : 1;
  guint                  disable_http_decompression : 1;
};

enum {
  PROP_0,
  PROP_PATH,
  PROP_URL,
  PROP_MD5,
  PROP_SHA1,
  PROP_SHA256,
  PROP_SHA512,
  PROP_STRIP_COMPONENTS,
  PROP_DEST_FILENAME,
  PROP_MIRROR_URLS,
  PROP_GIT_INIT,
  PROP_ARCHIVE_TYPE,
  PROP_HTTP_REFERER,
  PROP_DISABLE_HTTP_DECOMPRESSION,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakSourceArchive, foundry_flatpak_source_archive, FOUNDRY_TYPE_FLATPAK_SOURCE)

static void
foundry_flatpak_source_archive_finalize (GObject *object)
{
  FoundryFlatpakSourceArchive *self = (FoundryFlatpakSourceArchive *)object;

  g_clear_pointer (&self->archive_type, g_free);
  g_clear_pointer (&self->dest_filename, g_free);
  g_clear_pointer (&self->http_referer, g_free);
  g_clear_pointer (&self->md5, g_free);
  g_clear_pointer (&self->mirror_urls, g_strfreev);
  g_clear_pointer (&self->path, g_free);
  g_clear_pointer (&self->sha1, g_free);
  g_clear_pointer (&self->sha256, g_free);
  g_clear_pointer (&self->sha512, g_free);
  g_clear_pointer (&self->url, g_free);

  G_OBJECT_CLASS (foundry_flatpak_source_archive_parent_class)->finalize (object);
}

static void
foundry_flatpak_source_archive_get_property (GObject    *object,
                                             guint       prop_id,
                                             GValue     *value,
                                             GParamSpec *pspec)
{
  FoundryFlatpakSourceArchive *self = FOUNDRY_FLATPAK_SOURCE_ARCHIVE (object);

  switch (prop_id)
    {
    case PROP_PATH:
      g_value_set_string (value, self->path);
      break;

    case PROP_URL:
      g_value_set_string (value, self->url);
      break;

    case PROP_MD5:
      g_value_set_string (value, self->md5);
      break;

    case PROP_SHA1:
      g_value_set_string (value, self->sha1);
      break;

    case PROP_SHA256:
      g_value_set_string (value, self->sha256);
      break;

    case PROP_SHA512:
      g_value_set_string (value, self->sha512);
      break;

    case PROP_STRIP_COMPONENTS:
      g_value_set_uint (value, self->strip_components);
      break;

    case PROP_DEST_FILENAME:
      g_value_set_string (value, self->dest_filename);
      break;

    case PROP_MIRROR_URLS:
      g_value_set_boxed (value, self->mirror_urls);
      break;

    case PROP_GIT_INIT:
      g_value_set_boolean (value, self->git_init);
      break;

    case PROP_ARCHIVE_TYPE:
      g_value_set_string (value, self->archive_type);
      break;

    case PROP_HTTP_REFERER:
      g_value_set_string (value, self->http_referer);
      break;

    case PROP_DISABLE_HTTP_DECOMPRESSION:
      g_value_set_boolean (value, self->disable_http_decompression);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_archive_set_property (GObject      *object,
                                             guint         prop_id,
                                             const GValue *value,
                                             GParamSpec   *pspec)
{
  FoundryFlatpakSourceArchive *self = FOUNDRY_FLATPAK_SOURCE_ARCHIVE (object);

  switch (prop_id)
    {
    case PROP_PATH:
      g_set_str (&self->path, g_value_get_string (value));
      break;

    case PROP_URL:
      g_set_str (&self->url, g_value_get_string (value));
      break;

    case PROP_MD5:
      g_set_str (&self->md5, g_value_get_string (value));
      break;

    case PROP_SHA1:
      g_set_str (&self->sha1, g_value_get_string (value));
      break;

    case PROP_SHA256:
      g_set_str (&self->sha256, g_value_get_string (value));
      break;

    case PROP_SHA512:
      g_set_str (&self->sha512, g_value_get_string (value));
      break;

    case PROP_STRIP_COMPONENTS:
      self->strip_components = g_value_get_uint (value);
      break;

    case PROP_DEST_FILENAME:
      g_set_str (&self->dest_filename, g_value_get_string (value));
      break;

    case PROP_MIRROR_URLS:
      foundry_set_strv (&self->mirror_urls, g_value_get_boxed (value));
      break;

    case PROP_GIT_INIT:
      self->git_init = g_value_get_boolean (value);
      break;

    case PROP_ARCHIVE_TYPE:
      g_set_str (&self->archive_type, g_value_get_string (value));
      break;

    case PROP_HTTP_REFERER:
      g_set_str (&self->http_referer, g_value_get_string (value));
      break;

    case PROP_DISABLE_HTTP_DECOMPRESSION:
      self->disable_http_decompression = g_value_get_boolean (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_archive_class_init (FoundryFlatpakSourceArchiveClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryFlatpakSourceClass *source_class = FOUNDRY_FLATPAK_SOURCE_CLASS (klass);

  object_class->finalize = foundry_flatpak_source_archive_finalize;
  object_class->get_property = foundry_flatpak_source_archive_get_property;
  object_class->set_property = foundry_flatpak_source_archive_set_property;

  source_class->type = "archive";

  g_object_class_install_property (object_class,
                                   PROP_PATH,
                                   g_param_spec_string ("path",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_URL,
                                   g_param_spec_string ("url",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_MD5,
                                   g_param_spec_string ("md5",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_SHA1,
                                   g_param_spec_string ("sha1",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_SHA256,
                                   g_param_spec_string ("sha256",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_SHA512,
                                   g_param_spec_string ("sha512",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_STRIP_COMPONENTS,
                                   g_param_spec_uint ("strip-components",
                                                      NULL,
                                                      NULL,
                                                      0, G_MAXUINT,
                                                      1,
                                                      G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_DEST_FILENAME,
                                   g_param_spec_string ("dest-filename",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_MIRROR_URLS,
                                   g_param_spec_boxed ("mirror-urls",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_GIT_INIT,
                                   g_param_spec_boolean ("git-init",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_ARCHIVE_TYPE,
                                   g_param_spec_string ("archive-type",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_HTTP_REFERER,
                                   g_param_spec_string ("referer",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_DISABLE_HTTP_DECOMPRESSION,
                                   g_param_spec_boolean ("disable-http-decompression",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));
}

static void
foundry_flatpak_source_archive_init (FoundryFlatpakSourceArchive *self)
{
}
