/* foundry-text-document.c
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

#include "foundry-context.h"
#include "foundry-debug.h"
#include "foundry-diagnostic-manager.h"
#include "foundry-file-manager.h"
#include "foundry-on-type-diagnostics.h"
#include "foundry-operation.h"
#include "foundry-text-buffer-private.h"
#include "foundry-text-document-private.h"
#include "foundry-text-document-addin-private.h"
#include "foundry-text-edit.h"
#include "foundry-text-manager-private.h"
#include "foundry-text-settings-private.h"
#include "foundry-util-private.h"

struct _FoundryTextDocument
{
  FoundryContextual  parent_instance;
  GWeakRef           text_manager_wr;
  GWeakRef           on_type_diagnostics_wr;
  FoundryTextBuffer *buffer;
  GFile             *file;
  GIcon             *icon;
  char              *draft_id;
  PeasExtensionSet  *addins;
  DexPromise        *changed;
  DexFuture         *settings;
};

enum {
  PROP_0,
  PROP_ADDINS,
  PROP_BUFFER,
  PROP_DRAFT_ID,
  PROP_ICON,
  PROP_FILE,
  PROP_TITLE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryTextDocument, foundry_text_document, FOUNDRY_TYPE_CONTEXTUAL)

static GParamSpec *properties[N_PROPS];

static DexFuture *
foundry_text_document_update_icon_cb (DexFuture *completed,
                                      gpointer   data)
{
  FoundryTextDocument *self = data;
  g_autoptr(FoundryFileManager) file_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GFileInfo) file_info = NULL;
  g_autoptr(GIcon) icon = NULL;
  g_autofree char *basename = NULL;
  const char *content_type;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (FOUNDRY_IS_TEXT_DOCUMENT (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  file_manager = foundry_context_dup_file_manager (context);

  file_info = dex_await_object (dex_ref (completed), NULL);
  content_type = g_file_info_get_content_type (file_info);
  basename = g_file_get_basename (self->file);
  icon = foundry_file_manager_find_symbolic_icon (file_manager, content_type, basename);

  if (g_set_object (&self->icon, icon))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ICON]);

  return NULL;
}

static void
foundry_text_document_update_icon (FoundryTextDocument *self)
{
  g_assert (FOUNDRY_IS_TEXT_DOCUMENT (self));

  dex_future_disown (dex_future_then (dex_file_query_info (self->file,
                                                           G_FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE,
                                                           G_FILE_QUERY_INFO_NONE,
                                                           G_PRIORITY_DEFAULT),
                                      foundry_text_document_update_icon_cb,
                                      g_object_ref (self),
                                      g_object_unref));
}

static void
foundry_text_document_addin_added (PeasExtensionSet *set,
                                   PeasPluginInfo   *plugin_info,
                                   GObject          *object,
                                   gpointer          user_data)
{
  FoundryTextDocument *self = user_data;
  FoundryTextDocumentAddin *addin = (FoundryTextDocumentAddin *)object;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_TEXT_DOCUMENT_ADDIN (addin));
  g_assert (FOUNDRY_IS_TEXT_DOCUMENT (self));

  g_debug ("Adding FoundryTextDocumentAddin of type %s", G_OBJECT_TYPE_NAME (addin));

  _foundry_text_document_addin_set_document (addin, self);

  dex_future_disown (foundry_text_document_addin_load (FOUNDRY_TEXT_DOCUMENT_ADDIN (addin)));
}

static void
foundry_text_document_addin_removed (PeasExtensionSet *set,
                                     PeasPluginInfo   *plugin_info,
                                     GObject          *addin,
                                     gpointer          user_data)
{
  FoundryTextDocument *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_TEXT_DOCUMENT_ADDIN (addin));
  g_assert (FOUNDRY_IS_TEXT_DOCUMENT (self));

  g_debug ("Removing FoundryTextDocumentAddin of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_text_document_addin_unload (FOUNDRY_TEXT_DOCUMENT_ADDIN (addin)));
}

static DexFuture *
foundry_text_document_new_fiber (FoundryContext     *context,
                                 FoundryTextManager *text_manager,
                                 GFile              *file,
                                 const char         *draft_id,
                                 FoundryTextBuffer  *buffer)
{
  g_autoptr(FoundryTextDocument) self = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_CONTEXT (context));
  g_assert (FOUNDRY_IS_TEXT_MANAGER (text_manager));
  g_assert (G_IS_FILE (file));
  g_assert (FOUNDRY_IS_TEXT_BUFFER (buffer));

  self = g_object_new (FOUNDRY_TYPE_TEXT_DOCUMENT,
                       "context", context,
                       "file", file,
                       "draft-id", draft_id,
                       "buffer", buffer,
                       NULL);

  g_weak_ref_set (&self->text_manager_wr, text_manager);

  self->addins = peas_extension_set_new (peas_engine_get_default (),
                                         FOUNDRY_TYPE_TEXT_DOCUMENT_ADDIN,
                                         "context", context,
                                         NULL);
  g_signal_connect_object (self->addins,
                           "extension-added",
                           G_CALLBACK (foundry_text_document_addin_added),
                           self,
                           0);
  g_signal_connect_object (self->addins,
                           "extension-removed",
                           G_CALLBACK (foundry_text_document_addin_removed),
                           self,
                           0);

  _foundry_text_buffer_register (buffer, self);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryTextDocumentAddin) addin = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      _foundry_text_document_addin_set_document (addin, self);
      g_ptr_array_add (futures, foundry_text_document_addin_load (addin));
    }

  if (futures->len > 0)
    dex_await (foundry_future_all (futures), NULL);

  foundry_text_document_update_icon (self);

  return dex_future_new_take_object (g_steal_pointer (&self));
}

DexFuture *
_foundry_text_document_new (FoundryContext     *context,
                            FoundryTextManager *text_manager,
                            GFile              *file,
                            const char         *draft_id,
                            FoundryTextBuffer  *buffer)
{
  dex_return_error_if_fail (FOUNDRY_IS_CONTEXT (context));
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_MANAGER (text_manager));
  dex_return_error_if_fail (!file || G_IS_FILE (file));
  dex_return_error_if_fail (file != NULL || draft_id != NULL);
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_BUFFER (buffer));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_text_document_new_fiber),
                                  5,
                                  FOUNDRY_TYPE_CONTEXT, context,
                                  FOUNDRY_TYPE_TEXT_MANAGER, text_manager,
                                  G_TYPE_FILE, file,
                                  G_TYPE_STRING, draft_id,
                                  FOUNDRY_TYPE_TEXT_BUFFER, buffer);
}

static void
foundry_text_document_dispose (GObject *object)
{
  FoundryTextDocument *self = (FoundryTextDocument *)object;
  g_autoptr(FoundryTextManager) text_manager = NULL;

  if ((text_manager = g_weak_ref_get (&self->text_manager_wr)))
    _foundry_text_manager_release (text_manager, self);

  if (self->changed != NULL)
    {
      dex_promise_reject (self->changed,
                          g_error_new_literal (G_IO_ERROR,
                                               G_IO_ERROR_CANCELLED,
                                               "Object disposed"));
      dex_clear (&self->changed);
    }

  dex_clear (&self->settings);

  g_clear_object (&self->addins);

  G_OBJECT_CLASS (foundry_text_document_parent_class)->dispose (object);
}

static void
foundry_text_document_finalize (GObject *object)
{
  FoundryTextDocument *self = (FoundryTextDocument *)object;

  if (self->buffer != NULL)
    _foundry_text_buffer_unregister (self->buffer, self);

  g_clear_object (&self->buffer);
  g_clear_object (&self->icon);
  g_clear_object (&self->file);
  g_clear_pointer (&self->draft_id, g_free);
  g_weak_ref_clear (&self->text_manager_wr);
  g_weak_ref_clear (&self->on_type_diagnostics_wr);

  G_OBJECT_CLASS (foundry_text_document_parent_class)->finalize (object);
}

static void
foundry_text_document_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundryTextDocument *self = FOUNDRY_TEXT_DOCUMENT (object);

  switch (prop_id)
    {
    case PROP_ADDINS:
      g_value_take_object (value, foundry_text_document_list_addins (self));
      break;

    case PROP_BUFFER:
      g_value_take_object (value, foundry_text_document_dup_buffer (self));
      break;

    case PROP_DRAFT_ID:
      g_value_set_string (value, self->draft_id);
      break;

    case PROP_FILE:
      g_value_take_object (value, foundry_text_document_dup_file (self));
      break;

    case PROP_ICON:
      g_value_take_object (value, foundry_text_document_dup_icon (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_text_document_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_text_document_set_property (GObject      *object,
                                    guint         prop_id,
                                    const GValue *value,
                                    GParamSpec   *pspec)
{
  FoundryTextDocument *self = FOUNDRY_TEXT_DOCUMENT (object);

  switch (prop_id)
    {
    case PROP_BUFFER:
      self->buffer = g_value_dup_object (value);
      break;

    case PROP_FILE:
      self->file = g_value_dup_object (value);
      break;

    case PROP_DRAFT_ID:
      self->draft_id = g_value_dup_string (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_text_document_class_init (FoundryTextDocumentClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_text_document_dispose;
  object_class->finalize = foundry_text_document_finalize;
  object_class->get_property = foundry_text_document_get_property;
  object_class->set_property = foundry_text_document_set_property;

  properties[PROP_ADDINS] =
    g_param_spec_object ("addins", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_BUFFER] =
    g_param_spec_object ("buffer", NULL, NULL,
                         FOUNDRY_TYPE_TEXT_BUFFER,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DRAFT_ID] =
    g_param_spec_string ("draft-id", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_FILE] =
    g_param_spec_object ("file", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ICON] =
    g_param_spec_object ("icon", NULL, NULL,
                         G_TYPE_ICON,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_text_document_init (FoundryTextDocument *self)
{
  g_weak_ref_init (&self->text_manager_wr, NULL);
  g_weak_ref_init (&self->on_type_diagnostics_wr, NULL);
}

/**
 * foundry_text_document_dup_file:
 * @self: a #FoundryTextDocument
 *
 * Gets the underlying [iface@Gio.File].
 *
 * Returns: (transfer full) (nullable): a [iface@Gio.File] or %NULL
 */
GFile *
foundry_text_document_dup_file (FoundryTextDocument *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self), NULL);

  return self->file ? g_object_ref (self->file) : NULL;
}

/**
 * foundry_text_document_dup_buffer:
 * @self: a #FoundryTextDocument
 *
 * Gets the underlying [iface@Foundry.TextBuffer].
 *
 * Returns: (transfer full): a [iface@Foundry.TextBuffer].
 */
FoundryTextBuffer *
foundry_text_document_dup_buffer (FoundryTextDocument *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self), NULL);

  return self->buffer ? g_object_ref (self->buffer) : NULL;
}

/**
 * foundry_text_document_dup_title:
 * @self: a #FoundryTextDocument
 *
 * Gets the title for the document.
 *
 * Returns: (transfer full) (nullable): a string or %NULL
 */
char *
foundry_text_document_dup_title (FoundryTextDocument *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self), NULL);

  if (self->file)
    return g_file_get_basename (self->file);

  return NULL;
}

/**
 * foundry_text_document_when_changed:
 * @self: a #FoundryTextDocument
 *
 * This function will resolve a new [class@Dex.Future] when the underlying
 * [iface@Foundry.TextBuffer] has changed.
 *
 * Use this to invalidate external resources that rely on the contents of
 * the document buffer.
 *
 * This notification is done after a short delay to avoid spurious events
 * during rapid modification to the buffer.
 *
 * If the document is dispoed, this future will reject with error.
 *
 * Returns: (transfer full): a [class@Dex.Future]
 */
DexFuture *
foundry_text_document_when_changed (FoundryTextDocument *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self));

  if (self->changed == NULL)
    self->changed = dex_promise_new ();

  return dex_ref (self->changed);
}

/**
 * foundry_text_document_list_code_actions:
 * @self: a #FoundryTextDocument
 *
 * Queries [class@Foundry.TextDocumentAddin] for actions that are
 * relevant to the document based on the current diagnostics.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] of [class@Foundry.CodeAction].
 */
DexFuture *
foundry_text_document_list_code_actions (FoundryTextDocument *self)
{
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self));

  if (self->addins == NULL)
    return foundry_future_new_not_supported ();

  futures = g_ptr_array_new_with_free_func (dex_unref);
  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryTextDocumentAddin) addin = g_list_model_get_item (G_LIST_MODEL (self->addins), i);
      g_ptr_array_add (futures, foundry_text_document_addin_list_code_actions (addin));
    }

  return _foundry_flatten_list_model_new_from_futures (futures);
}

/**
 * foundry_text_document_diagnose:
 * @self: a #FoundryTextDocument
 *
 * Queries [class@Foundry.DiagnosticProvider] for diagnostics that are
 * relevant to the document.
 *
 * This does a single-shot run of diagnostics.
 *
 * To monitor diagnostics for changes, use
 * [method@Foundry.TextDocument.watch_diagnostics] which will update as
 * the document changes.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] of [class@Foundry.Diagnostic].
 */
DexFuture *
foundry_text_document_diagnose (FoundryTextDocument *self)
{
  g_autoptr(FoundryDiagnosticManager) diag_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GBytes) contents = NULL;
  g_autofree char *language_id = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self));
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_BUFFER (self->buffer));
  dex_return_error_if_fail (G_IS_FILE (self->file));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  diag_manager = foundry_context_dup_diagnostic_manager (context);

  contents = foundry_text_buffer_dup_contents (self->buffer);
  language_id = foundry_text_buffer_dup_language_id (self->buffer);

  return foundry_diagnostic_manager_diagnose (diag_manager, self->file, contents, language_id);
}

/**
 * foundry_text_document_watch_diagnostics:
 * @self: a [class@Foundry.TextDocument]
 *
 * Watches the document for changes and updates diagnostics in the
 * resulting [class@Foundry.OnTypeDiagnostics] which is a [iface@Gio.ListModel].
 *
 * Returns: (transfer full): a [class@Foundry.OnTypeDiagnostics]
 */
FoundryOnTypeDiagnostics *
foundry_text_document_watch_diagnostics (FoundryTextDocument *self)
{
  g_autoptr(FoundryOnTypeDiagnostics) ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self), NULL);
  g_return_val_if_fail (FOUNDRY_IS_TEXT_BUFFER (self->buffer), NULL);
  g_return_val_if_fail (G_IS_FILE (self->file), NULL);

  if (!(ret = g_weak_ref_get (&self->on_type_diagnostics_wr)))
    {
      ret = foundry_on_type_diagnostics_new (self);
      g_weak_ref_set (&self->on_type_diagnostics_wr, ret);
    }

  return g_steal_pointer (&ret);
}

/**
 * foundry_text_document_list_symbols:
 * @self: a #FoundryTextDocument
 *
 * Queries [class@Foundry.SymbolProvider] for symbols that are
 * relevant to the document.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] of [class@Foundry.Symbol].
 */
DexFuture *
foundry_text_document_list_symbols (FoundryTextDocument *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self));

  /* TODO: */

  return NULL;
}

/**
 * foundry_text_document_dup_icon:
 * @self: a [class@Foundry.TextDocument]
 *
 * Returns: (transfer full) (nullable):
 */
GIcon *
foundry_text_document_dup_icon (FoundryTextDocument *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self), NULL);

  return self->icon ? g_object_ref (self->icon) : NULL;
}

gboolean
foundry_text_document_apply_edit (FoundryTextDocument *self,
                                  FoundryTextEdit     *edit)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self), FALSE);
  g_return_val_if_fail (FOUNDRY_IS_TEXT_EDIT (edit), FALSE);

  return foundry_text_document_apply_edits (self, &edit, 1);
}

static int
compare_edit (gconstpointer a,
              gconstpointer b,
              gpointer      data)
{
  FoundryTextEdit *edit_a = *(FoundryTextEdit * const *)a;
  FoundryTextEdit *edit_b = *(FoundryTextEdit * const *)b;
  int cmpval = foundry_text_edit_compare (edit_a, edit_b);

  /* Reverse sort */
  if (cmpval < 0)
    return 1;
  else if (cmpval > 0)
    return -1;
  else
    return 0;
}

gboolean
foundry_text_document_apply_edits (FoundryTextDocument  *self,
                                   FoundryTextEdit     **edits,
                                   guint                 n_edits)
{
  g_autofree FoundryTextEdit **sorted = NULL;

  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self), FALSE);
  g_return_val_if_fail (edits != NULL || n_edits == 0, FALSE);

  if (n_edits == 0)
    return TRUE;

  sorted = g_memdup2 (edits, sizeof (FoundryTextEdit *) * n_edits);
  g_sort_array (sorted, n_edits, sizeof (FoundryTextEdit *), compare_edit, NULL);

  for (guint i = 0; i < n_edits; i++)
    {
      if (!foundry_text_buffer_apply_edit (self->buffer, sorted[i]))
        return FALSE;
    }

  return TRUE;
}

void
_foundry_text_document_changed (FoundryTextDocument *self)
{
  g_autoptr(DexPromise) promise = NULL;

  g_return_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self));

  if ((promise = g_steal_pointer (&self->changed)))
    dex_promise_resolve_boolean (promise, TRUE);
}

/**
 * foundry_text_document_dup_uri:
 * @self: a [class@Foundry.TextDocument]
 *
 * Convenience function to get the URI of a file.
 *
 * Returns: (transfer full): a UTF-8 encoded string containing the URI
 */
char *
foundry_text_document_dup_uri (FoundryTextDocument *self)
{
  g_autoptr(GFile) file = NULL;

  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self), NULL);

  if ((file = foundry_text_document_dup_file (self)))
    return g_file_get_uri (file);

  return NULL;
}

/**
 * foundry_text_document_save:
 * @self: a [class@Foundry.TextDocument]
 * @operation: (nullable): an operation to update with progress
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_text_document_save (FoundryTextDocument *self,
                            FoundryOperation    *operation)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self));
  dex_return_error_if_fail (!operation || FOUNDRY_IS_OPERATION (operation));
  dex_return_error_if_fail (G_IS_FILE (self->file));

  return foundry_text_document_save_as (self, self->file, operation);
}

typedef struct _Save
{
  FoundryContext      *context;
  FoundryTextDocument *document;
  FoundryTextBuffer   *buffer;
  GFile               *file;
  FoundryOperation    *operation;
  GListModel          *addins;
} Save;

static void
save_free (Save *state)
{
  g_clear_object (&state->addins);
  g_clear_object (&state->operation);
  g_clear_object (&state->file);
  g_clear_object (&state->buffer);
  g_clear_object (&state->document);
  g_clear_object (&state->context);
  g_free (state);
}

static DexFuture *
foundry_text_document_save_as_fiber (gpointer data)
{
  Save *state = data;
  g_autoptr(FoundryTextBufferProvider) text_buffer_provider = NULL;
  g_autoptr(FoundryTextManager) text_manager = NULL;
  g_autoptr(GPtrArray) pre = NULL;
  g_autoptr(GPtrArray) post = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_CONTEXT (state->context));
  g_assert (FOUNDRY_IS_TEXT_DOCUMENT (state->document));
  g_assert (FOUNDRY_IS_TEXT_BUFFER (state->buffer));
  g_assert (G_IS_FILE (state->file));
  g_assert (!state->addins || G_IS_LIST_MODEL (state->addins));
  g_assert (!state->operation || FOUNDRY_IS_OPERATION (state->operation));

  text_manager = foundry_context_dup_text_manager (state->context);
  text_buffer_provider = _foundry_text_manager_dup_provider (text_manager);

  /* Pre save phase */
  pre = g_ptr_array_new_with_free_func (dex_unref);
  for (guint i = 0; i < g_list_model_get_n_items (state->addins); i++)
    {
      g_autoptr(FoundryTextDocumentAddin) addin = g_list_model_get_item (state->addins, i);
      g_ptr_array_add (pre, foundry_text_document_addin_pre_save (addin));
    }

  if (pre->len > 0)
    dex_await (foundry_future_all (pre), NULL);

  /* Actual save operation */
  if (!dex_await (foundry_text_buffer_provider_save (text_buffer_provider,
                                                     state->buffer,
                                                     state->file,
                                                     state->operation,
                                                     NULL,  /* encoding */
                                                     NULL), /* CRLF */
                  &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  /* Post save phase */
  post = g_ptr_array_new_with_free_func (dex_unref);
  for (guint i = 0; i < g_list_model_get_n_items (state->addins); i++)
    {
      g_autoptr(FoundryTextDocumentAddin) addin = g_list_model_get_item (state->addins, i);
      g_ptr_array_add (post, foundry_text_document_addin_post_save (addin));
    }

  /* Clear settings as things may have changed */
  dex_clear (&state->document->settings);

  if (post->len > 0)
    dex_await (foundry_future_all (post), NULL);

  return dex_future_new_true ();
}

/**
 * foundry_text_document_save_as:
 * @self: a [class@Foundry.TextDocument]
 * @file: a [iface@Gio.File] of where to save the document
 * @operation: (nullable): an operation to update with progress
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any value
 *   or rejects with error.
 */
DexFuture *
foundry_text_document_save_as (FoundryTextDocument *self,
                               GFile               *file,
                               FoundryOperation    *operation)
{
  Save *state;

  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self));
  dex_return_error_if_fail (G_IS_FILE (file));
  dex_return_error_if_fail (!operation || FOUNDRY_IS_OPERATION (operation));

  state = g_new0 (Save, 1);
  state->context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  state->document = g_object_ref (self);
  state->buffer = g_object_ref (self->buffer);
  state->file = g_object_ref (file);
  state->operation = operation ? g_object_ref (operation) : NULL;
  state->addins = self->addins ? g_object_ref (G_LIST_MODEL (self->addins)) : NULL;

  return dex_scheduler_spawn (NULL, 0,
                              foundry_text_document_save_as_fiber,
                              state,
                              (GDestroyNotify) save_free);
}

/**
 * foundry_text_document_list_addins:
 * @self: a [class@Foundry.TextDocument]
 *
 * Returns: (transfer full) (nullable):
 */
GListModel *
foundry_text_document_list_addins (FoundryTextDocument *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self), NULL);

  if (self->addins != NULL)
    return g_object_ref (G_LIST_MODEL (self->addins));

  return NULL;
}

DexFuture *
_foundry_text_document_pre_load (FoundryTextDocument *self)
{
  g_autoptr(GPtrArray) pre = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self));
  dex_return_error_if_fail (self->addins != NULL);

  pre = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < g_list_model_get_n_items (G_LIST_MODEL (self->addins)); i++)
    {
      g_autoptr(FoundryTextDocumentAddin) addin = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (pre, foundry_text_document_addin_pre_load (addin));
    }

  if (pre->len > 0)
    return foundry_future_all (pre);

  return dex_future_new_true ();
}

DexFuture *
_foundry_text_document_post_load (FoundryTextDocument *self)
{
  g_autoptr(GPtrArray) post = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self));
  dex_return_error_if_fail (self->addins != NULL);

  post = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < g_list_model_get_n_items (G_LIST_MODEL (self->addins)); i++)
    {
      g_autoptr(FoundryTextDocumentAddin) addin = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (post, foundry_text_document_addin_post_load (addin));
    }

  if (post->len > 0)
    return foundry_future_all (post);

  return dex_future_new_true ();
}

/**
 * foundry_text_document_load_settings:
 * @self: a [class@Foundry.TextDocument]
 *
 * Loads the settings for the document.
 *
 * Text editing applications should bind the properties of the resulting
 * [class@Foundry.TextSettings] to their document's text editor.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.TextSettings] or rejects with error.
 */
DexFuture *
foundry_text_document_load_settings (FoundryTextDocument *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self));

  if (self->settings == NULL)
    {
      self->settings = _foundry_text_settings_new (self);
      dex_future_disown (dex_ref (self->settings));
    }

  return dex_ref (self->settings);
}

/**
 * foundry_text_document_find_addin:
 * @self: a [class@Foundry.TextDocument]
 * @module_name: the name of the plugin module
 *
 * Gets a [class@Foundry.TextDocumentAddin] using the name of the plugin.
 * This is the value for "Module=" in the `.plugin` file.
 *
 * Returns: (transfer full) (nullable) (type Foundry.TextDocumentAddin):
 */
gpointer
foundry_text_document_find_addin (FoundryTextDocument *self,
                                  const char          *module_name)
{
  PeasPluginInfo *plugin_info;
  gpointer addin = NULL;

  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (self), NULL);
  g_return_val_if_fail (module_name != NULL, NULL);
  g_return_val_if_fail (self->addins != NULL, NULL);

  if ((plugin_info = peas_engine_get_plugin_info (peas_engine_get_default (), module_name)))
    {
      if ((addin = peas_extension_set_get_extension (self->addins, plugin_info)))
        g_object_ref (addin);
    }

  return addin;
}
