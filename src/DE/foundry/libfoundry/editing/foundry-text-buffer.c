/* foundry-text-buffer.c
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

#include "foundry-context.h"
#include "foundry-operation.h"
#include "foundry-text-buffer-private.h"
#include "foundry-text-document-private.h"
#include "foundry-text-edit.h"

#define FOUNDRY_TEXT_DOCUMENTS_KEY "FOUNDRY_TEXT_DOCUMENTS"

G_DEFINE_INTERFACE (FoundryTextBuffer, foundry_text_buffer, G_TYPE_OBJECT)

static void
foundry_text_buffer_default_init (FoundryTextBufferInterface *iface)
{
  g_object_interface_install_property (iface,
                                       g_param_spec_object ("context", NULL, NULL,
                                                            FOUNDRY_TYPE_CONTEXT,
                                                            (G_PARAM_READWRITE |
                                                             G_PARAM_CONSTRUCT_ONLY |
                                                             G_PARAM_STATIC_STRINGS)));

  g_object_interface_install_property (iface,
                                       g_param_spec_string ("language-id", NULL, NULL,
                                                            NULL,
                                                            (G_PARAM_READABLE |
                                                             G_PARAM_STATIC_STRINGS)));
}

/**
 * foundry_text_buffer_dup_contents:
 * @self: a #FoundryTextBuffer
 *
 * Gets the contents of the buffer as a [struct@GLib.Bytes].
 *
 * Returns: (transfer full) (nullable): a #GBytes or %NULL
 */
GBytes *
foundry_text_buffer_dup_contents (FoundryTextBuffer *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_BUFFER (self), NULL);

  return FOUNDRY_TEXT_BUFFER_GET_IFACE (self)->dup_contents (self);
}

/**
 * foundry_text_buffer_settle:
 * @self: a #FoundryTextBuffer
 *
 * Gets a #DexFuture that will resolve after short delay when changes
 * have completed.
 *
 * Returns: (transfer full): a #DexFuture
 */
DexFuture *
foundry_text_buffer_settle (FoundryTextBuffer *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_BUFFER (self));

  return FOUNDRY_TEXT_BUFFER_GET_IFACE (self)->settle (self);
}

gboolean
foundry_text_buffer_apply_edit (FoundryTextBuffer  *self,
                                FoundryTextEdit    *edit)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_BUFFER (self), FALSE);
  g_return_val_if_fail (FOUNDRY_IS_TEXT_EDIT (edit), FALSE);

  return FOUNDRY_TEXT_BUFFER_GET_IFACE (self)->apply_edit (self, edit);
}

void
foundry_text_buffer_get_start_iter (FoundryTextBuffer *self,
                                    FoundryTextIter   *iter)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_BUFFER (self));
  g_return_if_fail (iter != NULL);

  FOUNDRY_TEXT_BUFFER_GET_IFACE (self)->iter_init (self, iter);
}

static GPtrArray *
foundry_text_buffer_ref_documents (FoundryTextBuffer *self)
{
  GPtrArray *ar;

  if (!(ar = g_object_get_data (G_OBJECT (self), FOUNDRY_TEXT_DOCUMENTS_KEY)))
    {
      ar = g_ptr_array_new ();
      g_object_set_data_full (G_OBJECT (self),
                              FOUNDRY_TEXT_DOCUMENTS_KEY,
                              ar,
                              (GDestroyNotify) g_ptr_array_unref);
    }

  return g_ptr_array_ref (ar);
}

void
_foundry_text_buffer_register (FoundryTextBuffer   *self,
                               FoundryTextDocument *document)
{
  g_autoptr(GPtrArray) documents = NULL;

  g_return_if_fail (FOUNDRY_IS_TEXT_BUFFER (self));
  g_return_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (document));

  if ((documents = foundry_text_buffer_ref_documents (self)))
    g_ptr_array_add (documents, document);
}

void
_foundry_text_buffer_unregister (FoundryTextBuffer   *self,
                                 FoundryTextDocument *document)
{
  g_autoptr(GPtrArray) documents = NULL;

  g_return_if_fail (FOUNDRY_IS_TEXT_BUFFER (self));
  g_return_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (document));

  if ((documents = foundry_text_buffer_ref_documents (self)))
    g_ptr_array_remove (documents, document);
}

void
foundry_text_buffer_emit_changed (FoundryTextBuffer *self)
{
  g_autoptr(GPtrArray) documents = NULL;

  g_return_if_fail (FOUNDRY_IS_TEXT_BUFFER (self));

  if ((documents = foundry_text_buffer_ref_documents (self)))
    {
      for (guint i = 0; i < documents->len; i++)
        _foundry_text_document_changed (documents->pdata[i]);
    }
}

/**
 * foundry_text_buffer_dup_language_id:
 * @self: a [iface@Foundry.TextBuffer]
 *
 * Gets the GtkSourceView-style identifier for the language of the buffer
 * such as "c" or "js".
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_text_buffer_dup_language_id (FoundryTextBuffer *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_BUFFER (self), NULL);

  if (FOUNDRY_TEXT_BUFFER_GET_IFACE (self)->dup_language_id)
    return FOUNDRY_TEXT_BUFFER_GET_IFACE (self)->dup_language_id (self);

  return NULL;
}

/**
 * foundry_text_buffer_get_change_count:
 * @self: a [iface@Foundry.TextBuffer]
 *
 * Gets the number of changes that have occurred to @buffer.
 *
 * This is generally just a monotonic number.
 */
gint64
foundry_text_buffer_get_change_count (FoundryTextBuffer *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_BUFFER (self), 0);

  if (FOUNDRY_TEXT_BUFFER_GET_IFACE (self)->get_change_count)
    return FOUNDRY_TEXT_BUFFER_GET_IFACE (self)->get_change_count (self);

  return 0;
}
