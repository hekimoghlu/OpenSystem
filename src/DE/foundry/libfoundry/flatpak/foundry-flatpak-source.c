/* foundry-flatpak-source.c
 *
 * Copyright 2015 Red Hat, Inc
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

#include "foundry-flatpak-source-private.h"
#include "foundry-util.h"

G_DEFINE_ABSTRACT_TYPE (FoundryFlatpakSource, foundry_flatpak_source, FOUNDRY_TYPE_FLATPAK_SERIALIZABLE)

enum {
  PROP_0,
  PROP_DEST,
  PROP_ONLY_ARCHES,
  PROP_SKIP_ARCHES,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_flatpak_source_finalize (GObject *object)
{
  FoundryFlatpakSource *self = FOUNDRY_FLATPAK_SOURCE (object);

  g_clear_pointer (&self->dest, g_free);
  g_clear_pointer (&self->only_arches, g_strfreev);
  g_clear_pointer (&self->skip_arches, g_strfreev);

  G_OBJECT_CLASS (foundry_flatpak_source_parent_class)->finalize (object);
}

static void
foundry_flatpak_source_get_property (GObject    *object,
                                     guint       prop_id,
                                     GValue     *value,
                                     GParamSpec *pspec)
{
  FoundryFlatpakSource *self = FOUNDRY_FLATPAK_SOURCE (object);

  switch (prop_id)
    {
    case PROP_DEST:
      g_value_take_string (value, foundry_flatpak_source_dup_dest (self));
      break;

    case PROP_ONLY_ARCHES:
      g_value_take_boxed (value, foundry_flatpak_source_dup_only_arches (self));
      break;

    case PROP_SKIP_ARCHES:
      g_value_take_boxed (value, foundry_flatpak_source_dup_skip_arches (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_set_property (GObject      *object,
                                     guint         prop_id,
                                     const GValue *value,
                                     GParamSpec   *pspec)
{
  FoundryFlatpakSource *self = FOUNDRY_FLATPAK_SOURCE (object);

  switch (prop_id)
    {
    case PROP_DEST:
      foundry_flatpak_source_set_dest (self, g_value_get_string (value));
      break;

    case PROP_ONLY_ARCHES:
      foundry_flatpak_source_set_only_arches (self, g_value_get_boxed (value));
      break;

    case PROP_SKIP_ARCHES:
      foundry_flatpak_source_set_skip_arches (self, g_value_get_boxed (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_class_init (FoundryFlatpakSourceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_flatpak_source_finalize;
  object_class->get_property = foundry_flatpak_source_get_property;
  object_class->set_property = foundry_flatpak_source_set_property;

  properties[PROP_DEST] =
    g_param_spec_string ("dest", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ONLY_ARCHES] =
    g_param_spec_boxed ("only-arches", NULL, NULL,
                        G_TYPE_STRV,
                        (G_PARAM_READWRITE |
                         G_PARAM_EXPLICIT_NOTIFY |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_SKIP_ARCHES] =
    g_param_spec_boxed ("skip-arches", NULL, NULL,
                        G_TYPE_STRV,
                        (G_PARAM_READWRITE |
                         G_PARAM_EXPLICIT_NOTIFY |
                         G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_flatpak_source_init (FoundryFlatpakSource *self)
{
}

char *
foundry_flatpak_source_dup_dest (FoundryFlatpakSource *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_SOURCE (self), NULL);

  return g_strdup (self->dest);
}

void
foundry_flatpak_source_set_dest (FoundryFlatpakSource *self,
                                 const char           *dest)
{
  g_return_if_fail (FOUNDRY_IS_FLATPAK_SOURCE (self));

  if (g_set_str (&self->dest, dest))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_DEST]);
}

/**
 * foundry_flatpak_source_dup_only_arches:
 * @self: a [class@Foundry.FlatpakSource]
 *
 * Returns: (transfer full) (nullable):
 */
char **
foundry_flatpak_source_dup_only_arches (FoundryFlatpakSource *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_SOURCE (self), NULL);

  return g_strdupv (self->only_arches);
}

void
foundry_flatpak_source_set_only_arches (FoundryFlatpakSource *self,
                                        const char * const   *only_arches)
{
  g_return_if_fail (FOUNDRY_IS_FLATPAK_SOURCE (self));

  if (foundry_set_strv (&self->only_arches, only_arches))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ONLY_ARCHES]);
}

/**
 * foundry_flatpak_source_dup_skip_arches:
 * @self: a [class@Foundry.FlatpakSource]
 *
 * Returns: (transfer full) (nullable):
 */
char **
foundry_flatpak_source_dup_skip_arches (FoundryFlatpakSource *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_SOURCE (self), NULL);

  return g_strdupv (self->skip_arches);
}

void
foundry_flatpak_source_set_skip_arches (FoundryFlatpakSource *self,
                                        const char * const   *skip_arches)
{
  g_return_if_fail (FOUNDRY_IS_FLATPAK_SOURCE (self));

  if (foundry_set_strv (&self->skip_arches, skip_arches))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SKIP_ARCHES]);
}
