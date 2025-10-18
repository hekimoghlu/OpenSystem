/* foundry-on-type-diagnostics.c
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

#include "foundry-diagnostic.h"
#include "foundry-model-manager.h"
#include "foundry-on-type-diagnostics.h"
#include "foundry-text-document.h"
#include "foundry-util-private.h"

#define INTERVAL_USEC (G_USEC_PER_SEC/4)

struct _FoundryOnTypeDiagnostics
{
  GObject     parent_instance;
  GWeakRef    document_wr;
  DexPromise *disposed;
  GListModel *model;
  GArray     *cache;
  gulong      items_changed_handler;
};

static GType
foundry_on_type_diagnostics_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_DIAGNOSTIC;
}

static guint
foundry_on_type_diagnostics_get_n_items (GListModel *model)
{
  FoundryOnTypeDiagnostics *self = FOUNDRY_ON_TYPE_DIAGNOSTICS (model);

  if (self->model == NULL)
    return 0;

  return g_list_model_get_n_items (self->model);
}

static gpointer
foundry_on_type_diagnostics_get_item (GListModel *model,
                                      guint       position)
{
  FoundryOnTypeDiagnostics *self = FOUNDRY_ON_TYPE_DIAGNOSTICS (model);

  if (self->model == NULL)
    return NULL;

  return g_list_model_get_item (self->model, position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_on_type_diagnostics_get_item_type;
  iface->get_n_items = foundry_on_type_diagnostics_get_n_items;
  iface->get_item = foundry_on_type_diagnostics_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryOnTypeDiagnostics, foundry_on_type_diagnostics, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

typedef struct _CacheEntry
{
  guint line;
  guint index;
} CacheEntry;

static int
sort_by_line (gconstpointer a,
              gconstpointer b)
{
  const CacheEntry *ca = a;
  const CacheEntry *cb = b;

  if (ca->line < cb->line)
    return -1;
  else if (ca->line > cb->line)
    return 1;
  else
    return 0;
}

static void
foundry_on_type_diagnostics_ensure_cache (FoundryOnTypeDiagnostics *self)
{
  guint n_items;

  g_assert (FOUNDRY_IS_ON_TYPE_DIAGNOSTICS (self));

  if (self->cache != NULL || self->model == NULL)
    return;

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self));

  self->cache = g_array_new (FALSE, FALSE, sizeof (CacheEntry));
  g_array_set_size (self->cache, n_items);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDiagnostic) diagnostic = g_list_model_get_item (self->model, i);
      CacheEntry *entry = &g_array_index (self->cache, CacheEntry, i);

      entry->line = foundry_diagnostic_get_line (diagnostic);
      entry->index = i;
    }

  g_array_sort (self->cache, sort_by_line);
}

static void
foundry_on_type_diagnostics_dispose (GObject *object)
{
  FoundryOnTypeDiagnostics *self = (FoundryOnTypeDiagnostics *)object;

  if (dex_future_is_pending (DEX_FUTURE (self->disposed)))
    dex_promise_reject (self->disposed,
                        g_error_new_literal (G_IO_ERROR,
                                             G_IO_ERROR_FAILED,
                                             "Object disposed"));

  g_clear_signal_handler (&self->items_changed_handler, self->model);
  g_clear_object (&self->model);
  g_clear_pointer (&self->cache, g_array_unref);

  G_OBJECT_CLASS (foundry_on_type_diagnostics_parent_class)->dispose (object);
}

static void
foundry_on_type_diagnostics_finalize (GObject *object)
{
  FoundryOnTypeDiagnostics *self = (FoundryOnTypeDiagnostics *)object;

  dex_clear (&self->disposed);
  g_weak_ref_clear (&self->document_wr);

  G_OBJECT_CLASS (foundry_on_type_diagnostics_parent_class)->finalize (object);
}

static void
foundry_on_type_diagnostics_class_init (FoundryOnTypeDiagnosticsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_on_type_diagnostics_dispose;
  object_class->finalize = foundry_on_type_diagnostics_finalize;
}

static void
foundry_on_type_diagnostics_init (FoundryOnTypeDiagnostics *self)
{
  self->disposed = dex_promise_new ();

  dex_future_disown (dex_ref (DEX_FUTURE (self->disposed)));
}

static void
foundry_on_type_diagnostics_replace (FoundryOnTypeDiagnostics *self,
                                     GListModel               *model)
{
  guint old_n_items = 0;
  guint new_n_items = 0;

  g_assert (FOUNDRY_IS_ON_TYPE_DIAGNOSTICS (self));
  g_assert (!model || G_IS_LIST_MODEL (model));

  dex_await (foundry_list_model_await (model), NULL);

  if (self->model == model)
    return;

  g_clear_pointer (&self->cache, g_array_unref);

  if (self->model != NULL)
    {
      old_n_items = g_list_model_get_n_items (self->model);
      g_clear_signal_handler (&self->items_changed_handler, self->model);
      g_clear_object (&self->model);
    }

  if (model != NULL)
    {
      self->model = g_object_ref (model);
      self->items_changed_handler = g_signal_connect_object (model,
                                                             "items-changed",
                                                             G_CALLBACK (g_list_model_items_changed),
                                                             self,
                                                             G_CONNECT_SWAPPED);
      new_n_items = g_list_model_get_n_items (model);
    }

  if (old_n_items || new_n_items)
    g_list_model_items_changed (G_LIST_MODEL (self), 0, old_n_items, new_n_items);
}

static DexFuture *
foundry_on_type_diagnostics_monitor (gpointer data)
{
  GWeakRef *self_wr = data;

  g_assert (self_wr != NULL);

  for (;;)
    {
      g_autoptr(FoundryOnTypeDiagnostics) self = NULL;
      g_autoptr(FoundryTextDocument) document = NULL;
      g_autoptr(DexFuture) changed = NULL;
      g_autoptr(DexFuture) future = NULL;
      gint64 next_deadline = g_get_monotonic_time () + INTERVAL_USEC;
      const GValue *value;

      if (!(self = g_weak_ref_get (self_wr)))
        break;

      if (!(document = g_weak_ref_get (&self->document_wr)))
        break;

      /* First setup a future that will complete when the document changes.
       * Do this first for a known starting point _before_ quering.
       */
      changed = foundry_text_document_when_changed (document);

      /* Now create a future that resolves or rejects when the live
       * diagnostics is disposed (e.g. no longer necessary) or the list
       * of diagnostics has come in.
       */
      future = dex_future_first (dex_ref (self->disposed),
                                 foundry_text_document_diagnose (document),
                                 NULL);

      /* Now we release our full references while we await any
       * sort of results so that object disposal may occur.
       */
      g_clear_object (&self);
      g_clear_object (&document);

      g_assert (self == NULL);
      g_assert (document == NULL);

      /* Await disposal or diagnostic availability */
      if (!dex_await (dex_ref (future), NULL))
        break;

      /* If we have disposed, then short-circuit */
      if (!(self = g_weak_ref_get (self_wr)))
        break;

      /* If we got results, then propagate our updated diagnostic list */
      if ((value = dex_future_get_value (future, NULL)) &&
          G_VALUE_HOLDS (value, G_TYPE_LIST_MODEL))
        foundry_on_type_diagnostics_replace (self, g_value_get_object (value));

      /* Now create a future that will resolve or reject with our instance
       * disposes or there is another change to the document.
       */
      dex_clear (&future);
      future = dex_future_first (dex_ref (self->disposed),
                                 dex_ref (changed),
                                 NULL);

      /* We're done with our instance, so release it again to allow for
       * disposal of the instance.
       */
      g_clear_object (&self);

      g_assert (self == NULL);
      g_assert (document == NULL);

      /* If we disposed then we're done here */
      if (!dex_await (dex_ref (future), NULL))
        break;

      /* Delay a little bit in case we're going too fast. */
      if (g_get_monotonic_time () < next_deadline)
        dex_await (dex_timeout_new_deadline (next_deadline), NULL);
    }

  return dex_future_new_true ();
}

FoundryOnTypeDiagnostics *
foundry_on_type_diagnostics_new (FoundryTextDocument *document)
{
  g_autoptr(FoundryOnTypeDiagnostics) self = NULL;

  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (document), NULL);

  self = g_object_new (FOUNDRY_TYPE_ON_TYPE_DIAGNOSTICS, NULL);
  g_weak_ref_set (&self->document_wr, document);

  dex_future_disown (dex_scheduler_spawn (NULL, 0,
                                          foundry_on_type_diagnostics_monitor,
                                          foundry_weak_ref_new (self),
                                          (GDestroyNotify) foundry_weak_ref_free));

  return g_steal_pointer (&self);
}

/**
 * foundry_on_type_diagnostics_foreach_in_range:
 * @self: a [class@Foundry.OnTypeDiagnostics]
 * @first_line: limit diagnostics to this line or after
 * @last_line: limit diagnostics to this line or before
 * @callback: (scope call):
 * @callback_data:
 *
 * Calls @callback for each diagnostic found between the two lines.
 */
void
foundry_on_type_diagnostics_foreach_in_range (FoundryOnTypeDiagnostics            *self,
                                              guint                                first_line,
                                              guint                                last_line,
                                              FoundryOnTypeDiagnosticsForeachFunc  callback,
                                              gpointer                             callback_data)
{
  g_return_if_fail (FOUNDRY_IS_ON_TYPE_DIAGNOSTICS (self));
  g_return_if_fail (callback != NULL);

  foundry_on_type_diagnostics_ensure_cache (self);

  if (self->model == NULL || self->cache == NULL || self->cache->len == 0)
    return;

  /* TODO: Opportunity to bsearch here when the number of diagnostics
   * is large. But in most cases it's small and therefore just walking
   * the array is more cacheline friendly.
   */

  for (guint i = 0; i < self->cache->len; i++)
    {
      const CacheEntry *entry = &g_array_index (self->cache, CacheEntry, i);

      if (entry->line < first_line)
        continue;

      if (entry->line > last_line)
        break;

      {
        g_autoptr(FoundryDiagnostic) diagnostic = g_list_model_get_item (self->model, entry->index);

        if (diagnostic == NULL)
          break;

        callback (diagnostic, callback_data);
      }
    }
}
