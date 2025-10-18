/* foundry-build-flags.c
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

#include "foundry-build-flags.h"

struct _FoundryBuildFlags
{
  GObject parent_instance;
  char *directory;
  char **flags;
} FoundryBuildFlagsPrivate;

enum {
  PROP_0,
  PROP_FLAGS,
  PROP_DIRECTORY,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryBuildFlags, foundry_build_flags, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_build_flags_finalize (GObject *object)
{
  FoundryBuildFlags *self = (FoundryBuildFlags *)object;

  g_clear_pointer (&self->flags, g_strfreev);
  g_clear_pointer (&self->directory, g_free);

  G_OBJECT_CLASS (foundry_build_flags_parent_class)->finalize (object);
}

static void
foundry_build_flags_get_property (GObject    *object,
                                  guint       prop_id,
                                  GValue     *value,
                                  GParamSpec *pspec)
{
  FoundryBuildFlags *self = FOUNDRY_BUILD_FLAGS (object);

  switch (prop_id)
    {
    case PROP_FLAGS:
      g_value_take_boxed (value, foundry_build_flags_dup_flags (self));
      break;

    case PROP_DIRECTORY:
      g_value_take_string (value, foundry_build_flags_dup_directory (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_build_flags_set_property (GObject      *object,
                                  guint         prop_id,
                                  const GValue *value,
                                  GParamSpec   *pspec)
{
  FoundryBuildFlags *self = FOUNDRY_BUILD_FLAGS (object);

  switch (prop_id)
    {
    case PROP_FLAGS:
      self->flags = g_value_dup_boxed (value);
      break;

    case PROP_DIRECTORY:
      self->directory = g_value_dup_string (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_build_flags_class_init (FoundryBuildFlagsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_build_flags_finalize;
  object_class->get_property = foundry_build_flags_get_property;
  object_class->set_property = foundry_build_flags_set_property;

  properties[PROP_FLAGS] =
    g_param_spec_boxed ("flags", NULL, NULL,
                         G_TYPE_STRV,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DIRECTORY] =
    g_param_spec_string ("directory", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_build_flags_init (FoundryBuildFlags *self)
{
}

/**
 * foundry_build_flags_dup_flags:
 * @self: a #FoundryBuildFlags
 *
 * Returns: (transfer full) (nullable): an array of build flags or %NULL
 */
char **
foundry_build_flags_dup_flags (FoundryBuildFlags *self)
{
  g_return_val_if_fail (FOUNDRY_IS_BUILD_FLAGS (self), NULL);

  return g_strdupv (self->flags);
}

/**
 * foundry_build_flags_dup_directory:
 * @self: a #FoundryBuildFlags
 *
 * Returns: (transfer full) (nullable): the directory for the build or %NULL
 */
char *
foundry_build_flags_dup_directory (FoundryBuildFlags *self)
{
  g_return_val_if_fail (FOUNDRY_IS_BUILD_FLAGS (self), NULL);

  return g_strdup (self->directory);
}

FoundryBuildFlags *
foundry_build_flags_new (const char * const *flags,
                         const char         *directory)
{
  return g_object_new (FOUNDRY_TYPE_BUILD_FLAGS,
                       "flags", flags,
                       "directory", directory,
                       NULL);
}
