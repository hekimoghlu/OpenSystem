/* foundry-text-formatter.c
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

#include "foundry-text-buffer.h"
#include "foundry-text-document.h"
#include "foundry-text-formatter.h"
#include "foundry-util.h"

typedef struct
{
  GWeakRef document_wr;
} FoundryTextFormatterPrivate;

enum {
  PROP_0,
  PROP_BUFFER,
  PROP_DOCUMENT,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryTextFormatter, foundry_text_formatter, FOUNDRY_TYPE_CONTEXTUAL)

static GParamSpec *properties[N_PROPS];

static void
foundry_text_formatter_dispose (GObject *object)
{
  FoundryTextFormatter *self = (FoundryTextFormatter *)object;
  FoundryTextFormatterPrivate *priv = foundry_text_formatter_get_instance_private (self);

  g_weak_ref_set (&priv->document_wr, NULL);

  G_OBJECT_CLASS (foundry_text_formatter_parent_class)->dispose (object);
}

static void
foundry_text_formatter_finalize (GObject *object)
{
  FoundryTextFormatter *self = (FoundryTextFormatter *)object;
  FoundryTextFormatterPrivate *priv = foundry_text_formatter_get_instance_private (self);

  g_weak_ref_clear (&priv->document_wr);

  G_OBJECT_CLASS (foundry_text_formatter_parent_class)->finalize (object);
}

static void
foundry_text_formatter_get_property (GObject    *object,
                                     guint       prop_id,
                                     GValue     *value,
                                     GParamSpec *pspec)
{
  FoundryTextFormatter *self = FOUNDRY_TEXT_FORMATTER (object);

  switch (prop_id)
    {
    case PROP_BUFFER:
      g_value_take_object (value, foundry_text_formatter_dup_buffer (self));
      break;

    case PROP_DOCUMENT:
      g_value_take_object (value, foundry_text_formatter_dup_document (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_text_formatter_set_property (GObject      *object,
                                     guint         prop_id,
                                     const GValue *value,
                                     GParamSpec   *pspec)
{
  FoundryTextFormatter *self = FOUNDRY_TEXT_FORMATTER (object);
  FoundryTextFormatterPrivate *priv = foundry_text_formatter_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_DOCUMENT:
      g_weak_ref_set (&priv->document_wr, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_text_formatter_class_init (FoundryTextFormatterClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_text_formatter_dispose;
  object_class->finalize = foundry_text_formatter_finalize;
  object_class->get_property = foundry_text_formatter_get_property;
  object_class->set_property = foundry_text_formatter_set_property;

  properties[PROP_BUFFER] =
    g_param_spec_object ("buffer", NULL, NULL,
                         FOUNDRY_TYPE_TEXT_BUFFER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DOCUMENT] =
    g_param_spec_object ("document", NULL, NULL,
                         FOUNDRY_TYPE_TEXT_DOCUMENT,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_text_formatter_init (FoundryTextFormatter *self)
{
}

/**
 * foundry_text_formatter_dup_document:
 * @self: a [class@Foundry.TextFormatter]
 *
 * Returns: (transfer full):
 */
FoundryTextDocument *
foundry_text_formatter_dup_document (FoundryTextFormatter *self)
{
  FoundryTextFormatterPrivate *priv = foundry_text_formatter_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_TEXT_FORMATTER (self), NULL);

  return g_weak_ref_get (&priv->document_wr);
}

/**
 * foundry_text_formatter_dup_buffer:
 * @self: a [class@Foundry.TextFormatter]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryTextBuffer *
foundry_text_formatter_dup_buffer (FoundryTextFormatter *self)
{
  g_autoptr(FoundryTextDocument) document = NULL;

  g_return_val_if_fail (FOUNDRY_IS_TEXT_FORMATTER (self), NULL);

  if ((document = foundry_text_formatter_dup_document (self)))
    return foundry_text_document_dup_buffer (document);

  return NULL;
}

/**
 * foundry_text_formatter_format:
 * @self: a [class@Foundry.TextFormatter]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error
 */
DexFuture *
foundry_text_formatter_format (FoundryTextFormatter *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_FORMATTER (self));

  if (FOUNDRY_TEXT_FORMATTER_GET_CLASS (self)->format)
    return FOUNDRY_TEXT_FORMATTER_GET_CLASS (self)->format (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_text_formatter_format_range:
 * @self: a [class@Foundry.TextFormatter]
 * @begin: the start of the region to format
 * @end: the end of the region to format
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error
 */
DexFuture *
foundry_text_formatter_format_range (FoundryTextFormatter  *self,
                                     const FoundryTextIter *begin,
                                     const FoundryTextIter *end)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_FORMATTER (self));
  dex_return_error_if_fail (begin != NULL);
  dex_return_error_if_fail (end != NULL);

  if (FOUNDRY_TEXT_FORMATTER_GET_CLASS (self)->format_range)
    return FOUNDRY_TEXT_FORMATTER_GET_CLASS (self)->format_range (self, begin, end);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_text_formatter_can_format_range:
 * @self: a [class@Foundry.TextFormatter]
 *
 * Determines if the formatter can do range formatting of the document.
 *
 * Returns: %TRUE if the formatter can handle format_range requests
 */
gboolean
foundry_text_formatter_can_format_range (FoundryTextFormatter *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_FORMATTER (self), FALSE);

  if (FOUNDRY_TEXT_FORMATTER_GET_CLASS (self)->can_format_range)
    return FOUNDRY_TEXT_FORMATTER_GET_CLASS (self)->can_format_range (self);

  return FALSE;
}
