/* foundry-flatpak-source-svn.c
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

#include "foundry-flatpak-source-svn.h"
#include "foundry-flatpak-source-private.h"

struct _FoundryFlatpakSourceSvn
{
  FoundryFlatpakSource  parent_instance;
  char                 *url;
  char                 *revision;
};

enum {
  PROP_0,
  PROP_REVISION,
  PROP_URL,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakSourceSvn, foundry_flatpak_source_svn, FOUNDRY_TYPE_FLATPAK_SOURCE)

static void
foundry_flatpak_source_svn_finalize (GObject *object)
{
  FoundryFlatpakSourceSvn *self = (FoundryFlatpakSourceSvn *)object;

  g_clear_pointer (&self->url, g_free);
  g_clear_pointer (&self->revision, g_free);

  G_OBJECT_CLASS (foundry_flatpak_source_svn_parent_class)->finalize (object);
}

static void
foundry_flatpak_source_svn_get_property (GObject    *object,
                                         guint       prop_id,
                                         GValue     *value,
                                         GParamSpec *pspec)
{
  FoundryFlatpakSourceSvn *self = FOUNDRY_FLATPAK_SOURCE_SVN (object);

  switch (prop_id)
    {
    case PROP_URL:
      g_value_set_string (value, self->url);
      break;

    case PROP_REVISION:
      g_value_set_string (value, self->revision);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_svn_set_property (GObject      *object,
                                         guint         prop_id,
                                         const GValue *value,
                                         GParamSpec   *pspec)
{
  FoundryFlatpakSourceSvn *self = FOUNDRY_FLATPAK_SOURCE_SVN (object);

  switch (prop_id)
    {
    case PROP_URL:
      g_set_str (&self->url, g_value_get_string (value));
      break;

    case PROP_REVISION:
      g_set_str (&self->revision, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_svn_class_init (FoundryFlatpakSourceSvnClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryFlatpakSourceClass *source_class = FOUNDRY_FLATPAK_SOURCE_CLASS (klass);

  object_class->finalize = foundry_flatpak_source_svn_finalize;
  object_class->get_property = foundry_flatpak_source_svn_get_property;
  object_class->set_property = foundry_flatpak_source_svn_set_property;

  source_class->type = "svn";

  g_object_class_install_property (object_class,
                                   PROP_REVISION,
                                   g_param_spec_string ("revision",
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
foundry_flatpak_source_svn_init (FoundryFlatpakSourceSvn *self)
{
}
