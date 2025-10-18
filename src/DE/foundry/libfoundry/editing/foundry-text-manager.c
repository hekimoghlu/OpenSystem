/* foundry-text-manager.c
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

#include <libpeas.h>

#include "foundry-extension.h"
#include "foundry-inhibitor.h"
#include "foundry-operation.h"
#include "foundry-simple-text-buffer-provider.h"
#include "foundry-text-buffer.h"
#include "foundry-text-buffer-provider.h"
#include "foundry-text-document-private.h"
#include "foundry-text-edit.h"
#include "foundry-text-manager-private.h"
#include "foundry-service-private.h"
#include "foundry-util-private.h"

struct _FoundryTextManager
{
  FoundryService             parent_instance;
  FoundryTextBufferProvider *text_buffer_provider;
  GHashTable                *documents_by_file;
  GHashTable                *loading;
};

struct _FoundryTextManagerClass
{
  FoundryServiceClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryTextManager, foundry_text_manager, FOUNDRY_TYPE_SERVICE)

enum {
  DOCUMENT_ADDED,
  DOCUMENT_REMOVED,
  N_SIGNALS
};

static guint signals[N_SIGNALS];

static DexFuture *
foundry_text_manager_start (FoundryService *service)
{
  FoundryTextManager *self = (FoundryTextManager *)service;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryExtension) text_buffer_provider = NULL;

  g_assert (FOUNDRY_IS_TEXT_MANAGER (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  text_buffer_provider = foundry_extension_new (context,
                                                peas_engine_get_default (),
                                                FOUNDRY_TYPE_TEXT_BUFFER_PROVIDER,
                                                "Text-Buffer-Provider", "*");

  /* You can only setup the buffer provider once at startup since that is what
   * will get used for all buffers that get displayed in UI/etc. They need to
   * be paired with their display counterpart (so GtkTextBuffer/GtkTextView).
   */
  g_set_object (&self->text_buffer_provider,
                foundry_extension_get_extension (text_buffer_provider));

  if (self->text_buffer_provider == NULL)
    self->text_buffer_provider = foundry_simple_text_buffer_provider_new (context);

  g_debug ("%s using %s as buffer provider",
           G_OBJECT_TYPE_NAME (self),
           G_OBJECT_TYPE_NAME (self->text_buffer_provider));

  return dex_future_new_true ();
}

static DexFuture *
foundry_text_manager_stop (FoundryService *service)
{
  FoundryTextManager *self = (FoundryTextManager *)service;

  g_assert (FOUNDRY_IS_TEXT_MANAGER (self));

  g_hash_table_remove_all (self->documents_by_file);
  g_hash_table_remove_all (self->loading);

  return dex_future_new_true ();
}

static void
foundry_text_manager_finalize (GObject *object)
{
  FoundryTextManager *self = (FoundryTextManager *)object;

  g_clear_pointer (&self->documents_by_file, g_hash_table_unref);
  g_clear_pointer (&self->loading, g_hash_table_unref);

  G_OBJECT_CLASS (foundry_text_manager_parent_class)->finalize (object);
}

static void
foundry_text_manager_class_init (FoundryTextManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->finalize = foundry_text_manager_finalize;

  service_class->start = foundry_text_manager_start;
  service_class->stop = foundry_text_manager_stop;

  signals[DOCUMENT_ADDED] =
    g_signal_new ("document-added",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  NULL,
                  G_TYPE_NONE, 2, G_TYPE_FILE, FOUNDRY_TYPE_TEXT_DOCUMENT);

  signals[DOCUMENT_REMOVED] =
    g_signal_new ("document-removed",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  NULL,
                  G_TYPE_NONE, 1, G_TYPE_FILE);
}

static void
foundry_text_manager_init (FoundryTextManager *self)
{
  self->documents_by_file = g_hash_table_new_full ((GHashFunc) g_file_hash,
                                                   (GEqualFunc) g_file_equal,
                                                   g_object_unref,
                                                   NULL);
  self->loading = g_hash_table_new_full ((GHashFunc) g_file_hash,
                                         (GEqualFunc) g_file_equal,
                                         g_object_unref,
                                         dex_unref);
}

static DexFuture *
foundry_text_manager_load_fiber (FoundryTextManager *self,
                                 GFile              *file,
                                 FoundryOperation   *operation,
                                 const char         *encoding)
{
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(FoundryTextBuffer) buffer = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(DexPromise) promise = NULL;
  g_autoptr(DexFuture) future = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *draft_id = NULL;
  FoundryTextDocument *existing;
  g_autofree char *uri = NULL;

  g_assert (FOUNDRY_IS_TEXT_MANAGER (self));
  g_assert (G_IS_FILE (file));
  g_assert (FOUNDRY_IS_OPERATION (operation));

  /* If loaded already, share the existing document */
  if ((existing = g_hash_table_lookup (self->documents_by_file, file)))
    return dex_future_new_take_object (g_object_ref (existing));

  /* If actively loading, await the same future */
  if ((future = g_hash_table_lookup (self->loading, file)))
    return dex_ref (future);

  promise = dex_promise_new ();
  g_hash_table_replace (self->loading, g_file_dup (file), dex_ref (promise));

  uri = g_file_get_uri (file);

  /* TODO: Stable draft-id */
  draft_id = NULL;

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  buffer = foundry_text_buffer_provider_create_buffer (self->text_buffer_provider);

  if (!(document = dex_await_object (_foundry_text_document_new (context, self, file, draft_id, buffer), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  dex_await (_foundry_text_document_pre_load (document), NULL);

  if (!dex_await (foundry_text_buffer_provider_load (self->text_buffer_provider,
                                                     buffer, file, operation,
                                                     encoding, NULL), &error))
    {
      g_hash_table_remove (self->loading, file);
      dex_promise_reject (promise, g_error_copy (error));
      return dex_future_new_for_error (g_steal_pointer (&error));
    }

  dex_await (_foundry_text_document_post_load (document), NULL);

  g_hash_table_remove (self->loading, file);

  /* Use borrowed reference, we'll get a callback to remove when
   * the document is disposed.
   */
  g_hash_table_replace (self->documents_by_file,
                        g_file_dup (file),
                        document);

  g_signal_emit (self, signals[DOCUMENT_ADDED], 0, file, document);

  dex_promise_resolve_object (promise, g_object_ref (document));

  return dex_future_new_take_object (g_steal_pointer (&document));
}

/**
 * foundry_text_manager_load:
 * @self: a #FoundryTextManager
 * @file: a #GFile
 * @operation: an operation for progress
 * @encoding: (nullable): an optional encoding charset
 *
 * Loads the file as a text document.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.TextDocument].
 */
DexFuture *
foundry_text_manager_load (FoundryTextManager *self,
                           GFile              *file,
                           FoundryOperation   *operation,
                           const char         *encoding)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_MANAGER (self));
  dex_return_error_if_fail (G_IS_FILE (file));
  dex_return_error_if_fail (FOUNDRY_IS_OPERATION (operation));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_text_manager_load_fiber),
                                  4,
                                  FOUNDRY_TYPE_TEXT_MANAGER, self,
                                  G_TYPE_FILE, file,
                                  FOUNDRY_TYPE_OPERATION, operation,
                                  G_TYPE_STRING, encoding);
}

/**
 * foundry_text_manager_list_documents:
 * @self: a [class@Foundry.TextManager]
 *
 * Returns: (transfer full): a [iface@Gio.ListModel] of [class@Foundry.TextDocument]
 */
GListModel *
foundry_text_manager_list_documents (FoundryTextManager *self)
{
  FoundryTextDocument *document;
  GHashTableIter iter;
  GListStore *store;

  g_return_val_if_fail (FOUNDRY_IS_TEXT_MANAGER (self), NULL);

  store = g_list_store_new (FOUNDRY_TYPE_TEXT_DOCUMENT);

  g_hash_table_iter_init (&iter, self->documents_by_file);
  while (g_hash_table_iter_next (&iter, NULL, (gpointer *)&document))
    g_list_store_append (store, document);

  return G_LIST_MODEL (store);
}

FoundryTextBufferProvider *
_foundry_text_manager_dup_provider (FoundryTextManager *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_MANAGER (self), NULL);

  return g_object_ref (self->text_buffer_provider);
}

typedef struct _ApplyEdits
{
  FoundryTextManager *self;
  FoundryOperation *operation;
  GHashTable *by_file;
} ApplyEdits;

static void
apply_edits_free (ApplyEdits *state)
{
  g_clear_object (&state->self);
  g_clear_object (&state->operation);
  g_clear_pointer (&state->by_file, g_hash_table_unref);
  g_free (state);
}

static DexFuture *
foundry_text_manager_apply_edits_fiber (gpointer data)
{
  ApplyEdits *state = data;
  GHashTableIter hiter;
  gpointer key, value;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_TEXT_MANAGER (state->self));
  g_assert (state->by_file != NULL);

  g_hash_table_iter_init (&hiter, state->by_file);

  while (g_hash_table_iter_next (&hiter, &key, &value))
    {
      g_autoptr(FoundryTextDocument) document = NULL;
      g_autoptr(GError) error = NULL;
      GPtrArray *edits = value;
      GFile *file = key;

      g_assert (G_IS_FILE (file));
      g_assert (edits != NULL);
      g_assert (edits->len > 0);

      if (!(document = dex_await_object (foundry_text_manager_load (state->self, file, state->operation, NULL), &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));

      if (!foundry_text_document_apply_edits (document, (FoundryTextEdit **)edits->pdata, edits->len))
        return dex_future_new_reject (G_IO_ERROR,
                                      G_IO_ERROR_INVALID_DATA,
                                      "Failed to apply edits to document");

    }

  return dex_future_new_true ();
}

/**
 * foundry_text_manager_apply_edits:
 * @self: a [class@Foundry.TextManager]
 * @edits: a [iface@Gio.ListModel] of [class@Foundry.TextEdit]
 * @operation: (nullable): a [class@Foundry.Operation]
 *
 * Applies all of @edits to the respective files.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error.
 */
DexFuture *
foundry_text_manager_apply_edits (FoundryTextManager *self,
                                  GListModel         *edits,
                                  FoundryOperation   *operation)
{
  g_autoptr(GHashTable) by_file = NULL;
  ApplyEdits *state;
  guint n_items;

  dex_return_error_if_fail (FOUNDRY_IS_TEXT_MANAGER (self));
  dex_return_error_if_fail (G_IS_LIST_MODEL (edits));
  dex_return_error_if_fail (FOUNDRY_IS_OPERATION (operation));

  n_items = g_list_model_get_n_items (edits);
  by_file = g_hash_table_new_full ((GHashFunc) g_file_hash,
                                   (GEqualFunc) g_file_equal,
                                   g_object_unref,
                                   (GDestroyNotify) g_ptr_array_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryTextEdit) edit = g_list_model_get_item (edits, i);
      g_autoptr(GFile) file = foundry_text_edit_dup_file (edit);
      GPtrArray *ar;

      g_assert (FOUNDRY_IS_TEXT_EDIT (edit));
      g_assert (G_IS_FILE (file));

      if (!(ar = g_hash_table_lookup (by_file, file)))
        {
          ar = g_ptr_array_new_with_free_func (g_object_unref);
          g_hash_table_replace (by_file, g_object_ref (file), ar);
        }

      g_ptr_array_add (ar, g_steal_pointer (&edit));
    }

  state = g_new0 (ApplyEdits, 1);
  state->self = g_object_ref (self);
  state->operation = operation ? g_object_ref (operation) : foundry_operation_new ();
  state->by_file = g_steal_pointer (&by_file);

  return dex_scheduler_spawn (NULL, 0,
                              foundry_text_manager_apply_edits_fiber,
                              state,
                              (GDestroyNotify) apply_edits_free);
}

void
_foundry_text_manager_release (FoundryTextManager  *self,
                               FoundryTextDocument *document)
{
  g_autoptr(GFile) file = NULL;
  GHashTableIter iter;
  gpointer value;

  g_return_if_fail (FOUNDRY_IS_MAIN_THREAD ());
  g_return_if_fail (FOUNDRY_IS_TEXT_MANAGER (self));
  g_return_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (document));

  g_debug ("Releasing text document at %p", document);

  file = foundry_text_document_dup_file (document);

  g_hash_table_iter_init (&iter, self->documents_by_file);
  while (g_hash_table_iter_next (&iter, NULL, &value))
    {
      if (value == (gpointer)document)
        {
          g_hash_table_iter_remove (&iter);
          break;
        }
    }

  g_signal_emit (self, signals[DOCUMENT_REMOVED], 0, file);
}
