/* foundry-flatpak-source-dir.c
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

#include "foundry-flatpak-source-dir.h"
#include "foundry-flatpak-source-private.h"
#include "foundry-util.h"

struct _FoundryFlatpakSourceDir
{
  FoundryFlatpakSource   parent_instance;
  char                  *path;
  char                 **skip;
};

enum {
  PROP_0,
  PROP_PATH,
  PROP_SKIP,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakSourceDir, foundry_flatpak_source_dir, FOUNDRY_TYPE_FLATPAK_SOURCE)

static void
foundry_flatpak_source_dir_finalize (GObject *object)
{
  FoundryFlatpakSourceDir *self = (FoundryFlatpakSourceDir *)object;

  g_clear_pointer (&self->path, g_free);
  g_clear_pointer (&self->skip, g_strfreev);

  G_OBJECT_CLASS (foundry_flatpak_source_dir_parent_class)->finalize (object);
}

static void
foundry_flatpak_source_dir_get_property (GObject    *object,
                                         guint       prop_id,
                                         GValue     *value,
                                         GParamSpec *pspec)
{
  FoundryFlatpakSourceDir *self = FOUNDRY_FLATPAK_SOURCE_DIR (object);

  switch (prop_id)
    {
    case PROP_PATH:
      g_value_set_string (value, self->path);
      break;

    case PROP_SKIP:
      g_value_set_boxed (value, self->skip);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_dir_set_property (GObject      *object,
                                         guint         prop_id,
                                         const GValue *value,
                                         GParamSpec   *pspec)
{
  FoundryFlatpakSourceDir *self = FOUNDRY_FLATPAK_SOURCE_DIR (object);

  switch (prop_id)
    {
    case PROP_PATH:
      g_set_str (&self->path, g_value_get_string (value));
      break;

    case PROP_SKIP:
      foundry_set_strv (&self->skip, g_value_get_boxed (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_dir_class_init (FoundryFlatpakSourceDirClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryFlatpakSourceClass *source_class = FOUNDRY_FLATPAK_SOURCE_CLASS (klass);

  object_class->finalize = foundry_flatpak_source_dir_finalize;
  object_class->get_property = foundry_flatpak_source_dir_get_property;
  object_class->set_property = foundry_flatpak_source_dir_set_property;

  source_class->type = "dir";

  g_object_class_install_property (object_class,
                                   PROP_PATH,
                                   g_param_spec_string ("path",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_SKIP,
                                   g_param_spec_boxed ("skip",
                                                        NULL,
                                                        NULL,
                                                        G_TYPE_STRV,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));
}

static void
foundry_flatpak_source_dir_init (FoundryFlatpakSourceDir *self)
{
}
