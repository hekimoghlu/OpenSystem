/* foundry-flatpak-source-patch.c
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

#include "foundry-flatpak-source-patch.h"
#include "foundry-flatpak-source-private.h"
#include "foundry-util.h"

struct _FoundryFlatpakSourcePatch
{
  FoundryFlatpakSource   parent_instance;

  char                  *path;

  char                 **paths;
  char                 **options;

  guint                  strip_components;

  guint                  use_git : 1;
  guint                  use_git_am : 1;
};

enum {
  PROP_0,
  PROP_PATH,
  PROP_PATHS,
  PROP_STRIP_COMPONENTS,
  PROP_USE_GIT,
  PROP_OPTIONS,
  PROP_USE_GIT_AM,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakSourcePatch, foundry_flatpak_source_patch, FOUNDRY_TYPE_FLATPAK_SOURCE)

static void
foundry_flatpak_source_patch_finalize (GObject *object)
{
  FoundryFlatpakSourcePatch *self = (FoundryFlatpakSourcePatch *)object;

  g_clear_pointer (&self->path, g_free);
  g_clear_pointer (&self->paths, g_strfreev);
  g_clear_pointer (&self->options, g_strfreev);

  G_OBJECT_CLASS (foundry_flatpak_source_patch_parent_class)->finalize (object);
}

static void
foundry_flatpak_source_patch_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  FoundryFlatpakSourcePatch *self = FOUNDRY_FLATPAK_SOURCE_PATCH (object);

  switch (prop_id)
    {
    case PROP_PATH:
      g_value_set_string (value, self->path);
      break;

    case PROP_PATHS:
      g_value_set_boxed (value, self->paths);
      break;

    case PROP_STRIP_COMPONENTS:
      g_value_set_uint (value, self->strip_components);
      break;

    case PROP_USE_GIT:
      g_value_set_boolean (value, self->use_git);
      break;

    case PROP_OPTIONS:
      g_value_set_boxed (value, self->options);
      break;

    case PROP_USE_GIT_AM:
      g_value_set_boolean (value, self->use_git_am);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_patch_set_property (GObject      *object,
                                           guint         prop_id,
                                           const GValue *value,
                                           GParamSpec   *pspec)
{
  FoundryFlatpakSourcePatch *self = FOUNDRY_FLATPAK_SOURCE_PATCH (object);

  switch (prop_id)
    {
    case PROP_PATH:
      g_set_str (&self->path, g_value_get_string (value));
      break;

    case PROP_PATHS:
      foundry_set_strv (&self->paths, g_value_get_boxed (value));
      break;

    case PROP_STRIP_COMPONENTS:
      self->strip_components = g_value_get_uint (value);
      break;

    case PROP_USE_GIT:
      self->use_git = g_value_get_boolean (value);
      break;

    case PROP_OPTIONS:
      foundry_set_strv (&self->options, g_value_get_boxed (value));
      break;

    case PROP_USE_GIT_AM:
      self->use_git_am = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_patch_class_init (FoundryFlatpakSourcePatchClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryFlatpakSourceClass *source_class = FOUNDRY_FLATPAK_SOURCE_CLASS (klass);

  object_class->finalize = foundry_flatpak_source_patch_finalize;
  object_class->get_property = foundry_flatpak_source_patch_get_property;
  object_class->set_property = foundry_flatpak_source_patch_set_property;

  source_class->type = "patch";

  g_object_class_install_property (object_class,
                                   PROP_PATH,
                                   g_param_spec_string ("path",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_PATHS,
                                   g_param_spec_boxed ("paths",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_STRIP_COMPONENTS,
                                   g_param_spec_uint ("strip-components",
                                                      NULL,
                                                      NULL,
                                                      0, G_MAXUINT,
                                                      1,
                                                      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_USE_GIT,
                                   g_param_spec_boolean ("use-git",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_OPTIONS,
                                   g_param_spec_boxed ("options",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_USE_GIT_AM,
                                   g_param_spec_boolean ("use-git-am",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

static void
foundry_flatpak_source_patch_init (FoundryFlatpakSourcePatch *self)
{
}
