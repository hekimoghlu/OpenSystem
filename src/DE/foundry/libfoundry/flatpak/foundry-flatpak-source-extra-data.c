/* foundry-flatpak-source-extra-data.c
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

#include "foundry-flatpak-source-extra-data.h"
#include "foundry-flatpak-source-private.h"

struct _FoundryFlatpakSourceExtraData
{
  FoundryFlatpakSource  parent_instance;
  char                 *filename;
  char                 *url;
  char                 *sha256;
};

enum {
  PROP_0,
  PROP_FILENAME,
  PROP_SHA256,
  PROP_URL,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakSourceExtraData, foundry_flatpak_source_extra_data, FOUNDRY_TYPE_FLATPAK_SOURCE)

static void
foundry_flatpak_source_extra_data_finalize (GObject *object)
{
  FoundryFlatpakSourceExtraData *self = (FoundryFlatpakSourceExtraData *)object;

  g_clear_pointer (&self->filename, g_free);
  g_clear_pointer (&self->sha256, g_free);
  g_clear_pointer (&self->url, g_free);

  G_OBJECT_CLASS (foundry_flatpak_source_extra_data_parent_class)->finalize (object);
}

static void
foundry_flatpak_source_extra_data_get_property (GObject    *object,
                                                guint       prop_id,
                                                GValue     *value,
                                                GParamSpec *pspec)
{
  FoundryFlatpakSourceExtraData *self = FOUNDRY_FLATPAK_SOURCE_EXTRA_DATA (object);

  switch (prop_id)
    {
    case PROP_FILENAME:
      g_value_set_string (value, self->filename);
      break;

    case PROP_SHA256:
      g_value_set_string (value, self->sha256);
      break;

    case PROP_URL:
      g_value_set_string (value, self->url);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_extra_data_set_property (GObject      *object,
                                                guint         prop_id,
                                                const GValue *value,
                                                GParamSpec   *pspec)
{
  FoundryFlatpakSourceExtraData *self = FOUNDRY_FLATPAK_SOURCE_EXTRA_DATA (object);

  switch (prop_id)
    {
    case PROP_FILENAME:
      g_set_str (&self->filename, g_value_get_string (value));
      break;

    case PROP_SHA256:
      g_set_str (&self->sha256, g_value_get_string (value));
      break;

    case PROP_URL:
      g_set_str (&self->url, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_extra_data_class_init (FoundryFlatpakSourceExtraDataClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryFlatpakSourceClass *source_class = FOUNDRY_FLATPAK_SOURCE_CLASS (klass);

  object_class->finalize = foundry_flatpak_source_extra_data_finalize;
  object_class->get_property = foundry_flatpak_source_extra_data_get_property;
  object_class->set_property = foundry_flatpak_source_extra_data_set_property;

  source_class->type = "sha256";

  g_object_class_install_property (object_class,
                                   PROP_FILENAME,
                                   g_param_spec_string ("filename",
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
                                   PROP_URL,
                                   g_param_spec_string ("url",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));
}

static void
foundry_flatpak_source_extra_data_init (FoundryFlatpakSourceExtraData *self)
{
}
