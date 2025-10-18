/* foundry-directory-listing.c
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

#include "libfoundry-config.h"

#include "foundry-directory-item-private.h"
#include "foundry-directory-listing.h"
#include "foundry-file-monitor.h"
#include "foundry-file-monitor-event.h"

#ifdef FOUNDRY_FEATURE_VCS
# include "foundry-vcs-manager.h"
# include "foundry-vcs.h"
#endif

#define VCS_IGNORED "vcs::ignored"
#define VCS_STATUS  "vcs::status"

struct _FoundryDirectoryListing
{
  FoundryContextual    parent_instance;
  DexPromise          *loaded;
  GSequence           *sequence;
  GHashTable          *file_to_item;
  GFile               *directory;
  char                *attributes;
  GFileQueryInfoFlags  query_flags;
  int                  num_files;
  int                  priority;
};

enum {
  PROP_0,
  PROP_ATTRIBUTES,
  PROP_DIRECTORY,
  PROP_QUERY_FLAGS,
  N_PROPS
};

static GType
foundry_directory_listing_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_DIRECTORY_ITEM;
}

static guint
foundry_directory_listing_get_n_items (GListModel *model)
{
  return g_sequence_get_length (FOUNDRY_DIRECTORY_LISTING (model)->sequence);
}

static gpointer
foundry_directory_listing_get_item (GListModel *model,
                                    guint       position)
{
  FoundryDirectoryListing *self = FOUNDRY_DIRECTORY_LISTING (model);
  GSequenceIter *iter = g_sequence_get_iter_at_pos (self->sequence, position);

  if (!g_sequence_iter_is_end (iter))
    return g_object_ref (g_sequence_get (iter));

  return NULL;
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_directory_listing_get_item_type;
  iface->get_n_items = foundry_directory_listing_get_n_items;
  iface->get_item = foundry_directory_listing_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryDirectoryListing, foundry_directory_listing, FOUNDRY_TYPE_CONTEXTUAL,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static GParamSpec *properties[N_PROPS];

static DexFuture *
foundry_directory_listing_fiber (gpointer data)
{
  g_autoptr(GFileEnumerator) enumerator = NULL;
  g_autoptr(FoundryFileMonitor) monitor = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(DexPromise) loaded = NULL;
  g_autoptr(GPtrArray) batch = NULL;
  g_autoptr(GFile) directory = NULL;
  g_autofree char *attributes = NULL;
  GFileQueryInfoFlags query_flags = 0;
  GWeakRef *wr = data;
  gboolean check_ignored = FALSE;
  gboolean check_status = FALSE;
  gpointer ptr;
  int num_files = 0;
  int prio = 0;

#ifdef FOUNDRY_FEATURE_VCS
  g_autoptr(FoundryVcsManager) vcs_manager = NULL;
  g_autoptr(FoundryVcs) vcs = NULL;
#endif

  /* First get our core objects we need persisted (without referencing
   * the @self object so it can get disposed while we run this fiber).
   *
   * We also create our file monitor which will queue changes until we
   * get to processing them (after we've enumerated the initial list).
   */
  if (!(ptr = g_weak_ref_get (wr)))
    {
      return dex_future_new_reject (G_IO_ERROR,
                                    G_IO_ERROR_CANCELLED,
                                    "Object disposed");
    }
  else
    {
      g_autoptr(FoundryDirectoryListing) self = ptr;
      g_autoptr(GError) error = NULL;

      directory = g_object_ref (self->directory);
      attributes = g_strdup (self->attributes);
      loaded = dex_ref (self->loaded);
      monitor = foundry_file_monitor_new (directory, NULL);
      query_flags = self->query_flags;
      num_files = self->num_files;
      prio = self->priority;

      if (!(enumerator = dex_await_object (dex_file_enumerate_children (directory, attributes, query_flags, prio),
                                           &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));

#ifdef FOUNDRY_FEATURE_VCS
      if ((context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))) &&
          (vcs_manager = foundry_context_dup_vcs_manager (context)) &&
          (vcs = foundry_vcs_manager_dup_vcs (vcs_manager)))
        {
          check_ignored = strstr (attributes, VCS_IGNORED) != NULL;
          check_status = strstr (attributes, VCS_STATUS) != NULL;
        }
#endif
    }

  batch = g_ptr_array_new_with_free_func (g_object_unref);

  /* Try to process the next block of results from the enumerator
   * but break out if our listing has been disposed while we are
   * processing those results.
   */
  while (enumerator != NULL)
    {
      g_autoptr(FoundryDirectoryListing) self = NULL;
      g_autolist(GFileInfo) files = NULL;
      g_autoptr(GError) error = NULL;
      guint position;
      guint added = 0;

      if (!(files = dex_await_boxed (dex_file_enumerator_next_files (enumerator, num_files, prio), &error)) ||
          !(self = g_weak_ref_get (wr)))
        {
          /* Cancel monitor if there was an error simply listing */
          if (monitor != NULL && error != NULL)
            foundry_file_monitor_cancel (monitor);
          break;
        }

      position = g_sequence_get_length (self->sequence);

      if ((guint64)position + num_files >= (guint64)G_MAXUINT)
        break;

      for (const GList *iter = files; iter; iter = iter->next)
        {
          GFileInfo *info = iter->data;
          g_autoptr(FoundryDirectoryItem) item = NULL;
          g_autoptr(GFile) file = g_file_enumerator_get_child (enumerator, info);

#ifdef FOUNDRY_FEATURE_VCS
          if (check_ignored)
            g_file_info_set_attribute_boolean (info,
                                               VCS_IGNORED,
                                               foundry_vcs_is_file_ignored (vcs, file));
          if (check_status)
            g_file_info_set_attribute_uint32 (info,
                                              VCS_STATUS,
                                              dex_await_flags (foundry_vcs_query_file_status (vcs, file), NULL));
#endif

          item = foundry_directory_item_new (directory, file, info);
          item->iter = g_sequence_append (self->sequence, g_object_ref (item));

          g_hash_table_insert (self->file_to_item,
                               g_steal_pointer (&file),
                               g_steal_pointer (&item));

          added++;
        }

      g_assert (added > 0);
      g_assert (G_MAXUINT - added > position);

      g_list_model_items_changed (G_LIST_MODEL (self), position, 0, added);
    }

  /* Now mark the loading process as complete if we didn't get disposed
   * (causing completion previously.
   */
  if (dex_future_is_pending (DEX_FUTURE (loaded)))
    dex_promise_resolve_boolean (loaded, TRUE);

  /* Now process queued changes from the monitor (and continue processing
   * new ones) until the listing is disposed or the monitor is cancelled
   * (which will happen during finalization).
   */
  if (monitor != NULL)
    {
      g_autoptr(GError) error = NULL;

      while ((ptr = dex_await_object (foundry_file_monitor_next (monitor), &error)))
        {
          g_autoptr(FoundryDirectoryListing) self = g_weak_ref_get (wr);
          g_autoptr(FoundryFileMonitorEvent) event = ptr;

          if (self == NULL)
            break;

          switch ((int)foundry_file_monitor_event_get_event (event))
            {
            case G_FILE_MONITOR_EVENT_CREATED:
              {
                g_autoptr(GFile) file = foundry_file_monitor_event_dup_file (event);
                g_autoptr(GFileInfo) info = NULL;

                if (g_hash_table_contains (self->file_to_item, file))
                  break;

                if ((info = dex_await_object (dex_file_query_info (file, attributes, query_flags, prio), NULL)))
                  {
                    g_autoptr(FoundryDirectoryItem) item = foundry_directory_item_new (directory, file, info);
                    guint position = g_sequence_get_length (self->sequence);
                    GSequenceIter *iter = g_sequence_append (self->sequence, g_object_ref (item));

                    item->iter = iter;

#ifdef FOUNDRY_FEATURE_VCS
                    if (check_ignored)
                      g_file_info_set_attribute_boolean (info,
                                                         VCS_IGNORED,
                                                         foundry_vcs_is_file_ignored (vcs, file));
                    if (check_status)
                      g_file_info_set_attribute_uint32 (info,
                                                        VCS_STATUS,
                                                        dex_await_flags (foundry_vcs_query_file_status (vcs, file), NULL));
#endif

                    g_hash_table_insert (self->file_to_item,
                                         g_steal_pointer (&file),
                                         g_steal_pointer (&item));

                    g_list_model_items_changed (G_LIST_MODEL (self), position, 0, 1);
                  }

                break;
              }

            case G_FILE_MONITOR_EVENT_DELETED:
              {
                g_autoptr(GFile) file = foundry_file_monitor_event_dup_file (event);
                FoundryDirectoryItem *item = g_hash_table_lookup (self->file_to_item, file);

                if (item != NULL)
                  {
                    GSequenceIter *iter = item->iter;
                    guint position = g_sequence_iter_get_position (iter);

                    item->iter = NULL;

                    g_sequence_remove (iter);
                    g_hash_table_remove (self->file_to_item, file);

                    g_list_model_items_changed (G_LIST_MODEL (self), position, 1, 0);
                  }

                break;
              }

            default:
              break;
            }
        }

      foundry_file_monitor_cancel (monitor);
    }

  return NULL;
}

static void
_g_weak_ref_free (GWeakRef *wr)
{
  g_weak_ref_clear (wr);
  g_free (wr);
}

static void
foundry_directory_listing_constructed (GObject *object)
{
  FoundryDirectoryListing *self = (FoundryDirectoryListing *)object;
  GWeakRef *wr;

  G_OBJECT_CLASS (foundry_directory_listing_parent_class)->constructed (object);

  wr = g_new0 (GWeakRef, 1);
  g_weak_ref_init (wr, self);

  dex_future_disown (dex_scheduler_spawn (NULL, 0,
                                          foundry_directory_listing_fiber,
                                          wr,
                                          (GDestroyNotify) _g_weak_ref_free));
}

static void
foundry_directory_listing_finalize (GObject *object)
{
  FoundryDirectoryListing *self = (FoundryDirectoryListing *)object;

  if (dex_future_is_pending (DEX_FUTURE (self->loaded)))
    dex_promise_reject (self->loaded,
                        g_error_new_literal (G_IO_ERROR,
                                             G_IO_ERROR_CANCELLED,
                                             "Listing disposed"));

  g_clear_pointer (&self->attributes, g_free);
  g_clear_object (&self->directory);
  dex_clear (&self->loaded);

  G_OBJECT_CLASS (foundry_directory_listing_parent_class)->finalize (object);
}

static void
foundry_directory_listing_get_property (GObject    *object,
                                        guint       prop_id,
                                        GValue     *value,
                                        GParamSpec *pspec)
{
  FoundryDirectoryListing *self = FOUNDRY_DIRECTORY_LISTING (object);

  switch (prop_id)
    {
    case PROP_ATTRIBUTES:
      g_value_set_string (value, self->attributes);
      break;

    case PROP_DIRECTORY:
      g_value_set_object (value, self->directory);
      break;

    case PROP_QUERY_FLAGS:
      g_value_set_flags (value, self->query_flags);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_directory_listing_set_property (GObject      *object,
                                        guint         prop_id,
                                        const GValue *value,
                                        GParamSpec   *pspec)
{
  FoundryDirectoryListing *self = FOUNDRY_DIRECTORY_LISTING (object);

  switch (prop_id)
    {
    case PROP_ATTRIBUTES:
      self->attributes = g_value_dup_string (value);
      break;

    case PROP_DIRECTORY:
      self->directory = g_value_dup_object (value);
      break;

    case PROP_QUERY_FLAGS:
      self->query_flags = g_value_get_flags (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_directory_listing_class_init (FoundryDirectoryListingClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->constructed = foundry_directory_listing_constructed;
  object_class->finalize = foundry_directory_listing_finalize;
  object_class->get_property = foundry_directory_listing_get_property;
  object_class->set_property = foundry_directory_listing_set_property;

  properties[PROP_ATTRIBUTES] =
    g_param_spec_string ("attributes", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DIRECTORY] =
    g_param_spec_object ("directory", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_QUERY_FLAGS] =
    g_param_spec_flags ("query-flags", NULL, NULL,
                        G_TYPE_FILE_QUERY_INFO_FLAGS,
                        0,
                        (G_PARAM_READWRITE |
                         G_PARAM_CONSTRUCT_ONLY |
                         G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_directory_listing_init (FoundryDirectoryListing *self)
{
  self->num_files = 100;
  self->priority = G_PRIORITY_DEFAULT;
  self->loaded = dex_promise_new ();
  self->sequence = g_sequence_new (g_object_unref);
  self->file_to_item = g_hash_table_new_full (g_file_hash,
                                              (GEqualFunc) g_file_equal,
                                              g_object_unref, g_object_unref);
}

FoundryDirectoryListing *
foundry_directory_listing_new (FoundryContext      *context,
                               GFile               *directory,
                               const char          *attributes,
                               GFileQueryInfoFlags  query_flags)
{
  g_return_val_if_fail (!context || FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (G_IS_FILE (directory), NULL);
  g_return_val_if_fail (attributes != NULL, NULL);

  return g_object_new (FOUNDRY_TYPE_DIRECTORY_LISTING,
                       "context", context,
                       "attributes", attributes,
                       "directory", directory,
                       "query-flags", query_flags,
                       NULL);
}

/**
 * foundry_directory_listing_await:
 * @self: a [class@Foundry.DirectoryListing]
 *
 * Await the completion of the initial directory listing.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a value or rejects with error.
 */
DexFuture *
foundry_directory_listing_await (FoundryDirectoryListing *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DIRECTORY_LISTING (self));

  return dex_ref (self->loaded);
}
