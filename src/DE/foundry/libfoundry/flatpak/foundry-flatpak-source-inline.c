/* foundry-flatpak-source-inline.c
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

#include "foundry-flatpak-source-inline.h"
#include "foundry-flatpak-source-private.h"

struct _FoundryFlatpakSourceInline
{
  FoundryFlatpakSource   parent_instance;
  char                  *contents;
  char                  *dest_filename;
  guint                  base64 : 1;
};

enum {
  PROP_0,
  PROP_CONTENTS,
  PROP_DEST_FILENAME,
  PROP_BASE64,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakSourceInline, foundry_flatpak_source_inline, FOUNDRY_TYPE_FLATPAK_SOURCE)

static void
foundry_flatpak_source_inline_finalize (GObject *object)
{
  FoundryFlatpakSourceInline *self = (FoundryFlatpakSourceInline *)object;

  g_clear_pointer (&self->contents, g_free);
  g_clear_pointer (&self->dest_filename, g_free);

  G_OBJECT_CLASS (foundry_flatpak_source_inline_parent_class)->finalize (object);
}

static void
foundry_flatpak_source_inline_get_property (GObject    *object,
                                            guint       prop_id,
                                            GValue     *value,
                                            GParamSpec *pspec)
{
  FoundryFlatpakSourceInline *self = FOUNDRY_FLATPAK_SOURCE_INLINE (object);

  switch (prop_id)
    {
    case PROP_CONTENTS:
      g_value_set_string (value, self->contents);
      break;

    case PROP_DEST_FILENAME:
      g_value_set_string (value, self->dest_filename);
      break;

    case PROP_BASE64:
      g_value_set_boolean (value, self->base64);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_inline_set_property (GObject      *object,
                                            guint         prop_id,
                                            const GValue *value,
                                            GParamSpec   *pspec)
{
  FoundryFlatpakSourceInline *self = FOUNDRY_FLATPAK_SOURCE_INLINE (object);

  switch (prop_id)
    {
    case PROP_CONTENTS:
      g_set_str (&self->contents, g_value_get_string (value));
      break;

    case PROP_DEST_FILENAME:
      g_set_str (&self->dest_filename, g_value_get_string (value));
      break;

    case PROP_BASE64:
      self->base64 = g_value_get_boolean (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_source_inline_class_init (FoundryFlatpakSourceInlineClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryFlatpakSourceClass *source_class = FOUNDRY_FLATPAK_SOURCE_CLASS (klass);

  object_class->finalize = foundry_flatpak_source_inline_finalize;
  object_class->get_property = foundry_flatpak_source_inline_get_property;
  object_class->set_property = foundry_flatpak_source_inline_set_property;

  source_class->type = "inline";

  g_object_class_install_property (object_class,
                                   PROP_CONTENTS,
                                   g_param_spec_string ("contents",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_DEST_FILENAME,
                                   g_param_spec_string ("dest-filename",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));

  g_object_class_install_property (object_class,
                                   PROP_BASE64,
                                   g_param_spec_boolean ("base64",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_STATIC_STRINGS | G_PARAM_READWRITE));
}

static void
foundry_flatpak_source_inline_init (FoundryFlatpakSourceInline *self)
{
}
