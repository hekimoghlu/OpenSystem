/* foundry-markup.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include "foundry-markup.h"

struct _FoundryMarkup
{
  GObject            parent_instance;
  GBytes            *contents;
  FoundryMarkupKind  kind;
};

G_DEFINE_ENUM_TYPE (FoundryMarkupKind, foundry_markup_kind,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_MARKUP_KIND_PLAINTEXT, "plaintext"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_MARKUP_KIND_MARKDOWN, "markdown"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_MARKUP_KIND_HTML, "html"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_MARKUP_KIND_PANGO, "pango"))

G_DEFINE_FINAL_TYPE (FoundryMarkup, foundry_markup, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_CONTENTS,
  PROP_KIND,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

FoundryMarkup *
foundry_markup_new (GBytes            *contents,
                    FoundryMarkupKind  kind)
{
  g_return_val_if_fail (contents != NULL, NULL);
  g_return_val_if_fail (kind <= FOUNDRY_MARKUP_KIND_PANGO, NULL);

  return g_object_new (FOUNDRY_TYPE_MARKUP,
                       "contents", contents,
                       "kind", kind,
                       NULL);
}

static void
foundry_markup_finalize (GObject *object)
{
  FoundryMarkup *self = (FoundryMarkup *)object;

  g_clear_pointer (&self->contents, g_bytes_unref);

  G_OBJECT_CLASS (foundry_markup_parent_class)->finalize (object);
}

static void
foundry_markup_get_property (GObject    *object,
                             guint       prop_id,
                             GValue     *value,
                             GParamSpec *pspec)
{
  FoundryMarkup *self = FOUNDRY_MARKUP (object);

  switch (prop_id)
    {
    case PROP_CONTENTS:
      g_value_take_boxed (value, foundry_markup_dup_contents (self));
      break;

    case PROP_KIND:
      g_value_set_enum (value, foundry_markup_get_kind (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_markup_set_property (GObject      *object,
                             guint         prop_id,
                             const GValue *value,
                             GParamSpec   *pspec)
{
  FoundryMarkup *self = FOUNDRY_MARKUP (object);

  switch (prop_id)
    {
    case PROP_CONTENTS:
      self->contents = g_value_dup_boxed (value);
      break;

    case PROP_KIND:
      self->kind = g_value_get_enum (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_markup_class_init (FoundryMarkupClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_markup_finalize;
  object_class->get_property = foundry_markup_get_property;
  object_class->set_property = foundry_markup_set_property;

  properties[PROP_CONTENTS] =
    g_param_spec_boxed ("contents", NULL, NULL,
                         G_TYPE_BYTES,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_KIND] =
    g_param_spec_enum ("kind", NULL, NULL,
                       FOUNDRY_TYPE_MARKUP_KIND,
                       FOUNDRY_MARKUP_KIND_PLAINTEXT,
                       (G_PARAM_READWRITE |
                        G_PARAM_CONSTRUCT_ONLY |
                        G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_markup_init (FoundryMarkup *self)
{
}

/**
 * foundry_markup_dup_contents:
 * @self: a #FoundryMarkup
 *
 * Gets the contents bytes.
 *
 * Returns: (transfer full): a #GBytes containing the markup
 */
GBytes *
foundry_markup_dup_contents (FoundryMarkup *self)
{
  g_return_val_if_fail (self != NULL, NULL);

  return g_bytes_ref (self->contents);
}

FoundryMarkupKind
foundry_markup_get_kind (FoundryMarkup *self)
{
  g_return_val_if_fail (self != NULL, 0);

  return self->kind;
}

FoundryMarkup *
foundry_markup_new_plaintext (const char *message)
{
  g_autoptr(GBytes) contents = NULL;

  g_return_val_if_fail (message != NULL, NULL);

  contents = g_bytes_new_take (g_strdup (message), strlen (message));

  return foundry_markup_new (contents, FOUNDRY_MARKUP_KIND_PLAINTEXT);
}

/**
 * foundry_markup_to_string:
 * @self: a [class@Foundry.Markup]
 * @length: (out): length of resulting string
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_markup_to_string (FoundryMarkup *self,
                          gsize         *length)
{
  g_return_val_if_fail (FOUNDRY_IS_MARKUP (self), NULL);

  if (self->contents == NULL)
    {
      if (length != NULL)
        *length = 0;

      return NULL;
    }
  else
    {
      if (length != NULL)
        *length = g_bytes_get_size (self->contents);

      return g_strndup (g_bytes_get_data (self->contents, NULL),
                        g_bytes_get_size (self->contents));
    }
}
