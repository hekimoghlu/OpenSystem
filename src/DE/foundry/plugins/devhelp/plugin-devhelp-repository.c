/* plugin-devhelp-repository.c
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

#include "foundry-gom-private.h"

#include "plugin-devhelp-book.h"
#include "plugin-devhelp-heading.h"
#include "plugin-devhelp-keyword.h"
#include "plugin-devhelp-repository.h"
#include "plugin-devhelp-sdk.h"

#define PLUGIN_DEVHELP_REPOSITORY_VERSION 1

struct _PluginDevhelpRepository
{
  GomRepository  parent_instance;
  GHashTable    *cached_book_titles;
  GHashTable    *cached_sdk_titles;
  GHashTable    *cached_book_to_sdk_id;
};

G_DEFINE_FINAL_TYPE (PluginDevhelpRepository, plugin_devhelp_repository, GOM_TYPE_REPOSITORY)

static void
plugin_devhelp_repository_finalize (GObject *object)
{
  PluginDevhelpRepository *self = (PluginDevhelpRepository *)object;

  g_clear_pointer (&self->cached_book_to_sdk_id, g_hash_table_unref);
  g_clear_pointer (&self->cached_book_titles, g_hash_table_unref);
  g_clear_pointer (&self->cached_sdk_titles, g_hash_table_unref);

  G_OBJECT_CLASS (plugin_devhelp_repository_parent_class)->finalize (object);
}

static void
plugin_devhelp_repository_class_init (PluginDevhelpRepositoryClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = plugin_devhelp_repository_finalize;
}

static void
plugin_devhelp_repository_init (PluginDevhelpRepository *self)
{
  self->cached_book_to_sdk_id = g_hash_table_new_full (g_int64_hash, g_int64_equal, g_free, g_free);
  self->cached_book_titles = g_hash_table_new_full (g_int64_hash, g_int64_equal, g_free, g_free);
  self->cached_sdk_titles = g_hash_table_new_full (g_int64_hash, g_int64_equal, g_free, g_free);
}

static DexFuture *
plugin_devhelp_repository_open_fiber (gpointer user_data)
{
  g_autoptr(GomAdapter) adapter = NULL;
  g_autoptr(PluginDevhelpRepository) self = NULL;
  g_autoptr(GError) error = NULL;
  const char *uri = user_data;
  GList *types = NULL;

  g_assert (uri != NULL);

  /* Open a sqlite connection to the file */
  adapter = gom_adapter_new ();
  if (!dex_await (gom_adapter_open (adapter, uri), &error))
    {
      dex_await (gom_adapter_close (adapter), NULL);
      return dex_future_new_for_error (g_steal_pointer (&error));
    }

  /* Create our repository using the sqlite adpater */
  self = g_object_new (PLUGIN_TYPE_DEVHELP_REPOSITORY,
                       "adapter", adapter,
                       NULL);

  /* Now make sure our migrations are ready */
  types = g_list_prepend (types, GSIZE_TO_POINTER (PLUGIN_TYPE_DEVHELP_KEYWORD));
  types = g_list_prepend (types, GSIZE_TO_POINTER (PLUGIN_TYPE_DEVHELP_HEADING));
  types = g_list_prepend (types, GSIZE_TO_POINTER (PLUGIN_TYPE_DEVHELP_BOOK));
  types = g_list_prepend (types, GSIZE_TO_POINTER (PLUGIN_TYPE_DEVHELP_SDK));
  if (!dex_await (gom_repository_automatic_migrate (GOM_REPOSITORY (self),
                                                    PLUGIN_DEVHELP_REPOSITORY_VERSION,
                                                    g_steal_pointer (&types)),
                  &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  /* We're ready, let the caller have the instance */
  return dex_future_new_for_object (g_steal_pointer (&self));
}

DexFuture *
plugin_devhelp_repository_open (const char *path)
{
  g_autoptr(GFile) file = NULL;

  g_return_val_if_fail (path != NULL, NULL);

  file = g_file_new_for_uri (path);

  return dex_scheduler_spawn (NULL, 0,
                              plugin_devhelp_repository_open_fiber,
                              g_file_get_uri (file),
                              g_free);
}

static DexFuture *
do_nothing_cb (DexFuture *completed,
               gpointer   user_data)
{
  return dex_ref (completed);
}

DexFuture *
plugin_devhelp_repository_close (PluginDevhelpRepository *self)
{
  GomAdapter *adapter;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self), NULL);

  if (!(adapter = gom_repository_get_adapter (GOM_REPOSITORY (self))))
    return dex_future_new_for_boolean (TRUE);

  return dex_future_finally (gom_adapter_close (adapter),
                             do_nothing_cb,
                             g_object_ref (self),
                             g_object_unref);
}

static void
plugin_devhelp_repository_find_one_cb (GObject      *object,
                                GAsyncResult *result,
                                gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GomResource) resource = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (PLUGIN_IS_DEVHELP_REPOSITORY (object));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_FUTURE (promise));

  resource = gom_repository_find_one_finish (GOM_REPOSITORY (object), result, &error);

  if (error != NULL)
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_object (promise, g_steal_pointer (&resource));
}

DexFuture *
plugin_devhelp_repository_find_one (PluginDevhelpRepository *self,
                                    GType              resource_type,
                                    GomFilter         *filter)
{
  DexPromise *promise;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self), NULL);
  g_return_val_if_fail (g_type_is_a (resource_type, GOM_TYPE_RESOURCE), NULL);
  g_return_val_if_fail (!filter || GOM_IS_FILTER (filter), NULL);

  promise = dex_promise_new ();
  gom_repository_find_one_async (GOM_REPOSITORY (self),
                                 resource_type,
                                 filter,
                                 plugin_devhelp_repository_find_one_cb,
                                 dex_ref (promise));
  return DEX_FUTURE (promise);
}

static DexFuture *
plugin_devhelp_repository_list_sdks_fiber (gpointer user_data)
{
  PluginDevhelpRepository *self = user_data;
  g_autoptr(GomResourceGroup) resource_group = NULL;
  g_autoptr(GListStore) list = NULL;
  g_autoptr(GError) error = NULL;
  GomSorting *sorting = NULL;
  DexFuture *sorted;
  guint count;

  g_assert (PLUGIN_IS_DEVHELP_REPOSITORY (self));

  sorting = gom_sorting_new (PLUGIN_TYPE_DEVHELP_SDK, "name", GOM_SORTING_ASCENDING,
                             PLUGIN_TYPE_DEVHELP_SDK, "version", GOM_SORTING_DESCENDING,
                             G_TYPE_INVALID);
  sorted = gom_repository_find_sorted (GOM_REPOSITORY (self),
                                       PLUGIN_TYPE_DEVHELP_SDK,
                                       NULL,
                                       sorting);
  g_clear_object (&sorting);

  if (!(resource_group = dex_await_object (sorted, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  count = gom_resource_group_get_count (resource_group);
  if (!dex_await (gom_resource_group_fetch (resource_group, 0, count), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  list = g_list_store_new (PLUGIN_TYPE_DEVHELP_SDK);
  for (guint i = 0; i < count; i++)
    {
      GomResource *resource = gom_resource_group_get_index (resource_group, i);

      if (resource != NULL)
        g_list_store_append (list, resource);
    }

  return dex_future_new_for_object (g_steal_pointer (&list));
}

DexFuture *
plugin_devhelp_repository_list_sdks (PluginDevhelpRepository *self)
{
  g_return_val_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self), NULL);

  return dex_scheduler_spawn (NULL, 0,
                              plugin_devhelp_repository_list_sdks_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static void
plugin_devhelp_repository_delete_cb (GomAdapter *adapter,
                                     gpointer    user_data)
{
  g_autoptr(GomCommand) command = user_data;
  g_autoptr(GError) error = NULL;
  DexPromise *promise;

  g_assert (GOM_IS_ADAPTER (adapter));
  g_assert (GOM_IS_COMMAND (command));

  promise = g_object_get_data (G_OBJECT (command), "DEX_PROMISE");

  g_assert (DEX_IS_PROMISE (promise));

  if (!gom_command_execute (command, NULL, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boolean (promise, TRUE);
}

DexFuture *
plugin_devhelp_repository_delete (PluginDevhelpRepository *self,
                                  GType                    resource_type,
                                  GomFilter               *filter)
{
  g_autoptr(GomCommandBuilder) builder = NULL;
  g_autoptr(GomCommand) command = NULL;
  DexPromise *promise;
  GomAdapter *adapter;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self), NULL);
  g_return_val_if_fail (g_type_is_a (resource_type, GOM_TYPE_RESOURCE), NULL);
  g_return_val_if_fail (GOM_IS_FILTER (filter), NULL);

  adapter = gom_repository_get_adapter (GOM_REPOSITORY (self));
  builder = g_object_new (GOM_TYPE_COMMAND_BUILDER,
                          "adapter", adapter,
                          "filter", filter,
                          "resource-type", resource_type,
                          NULL);
  command = gom_command_builder_build_delete (builder);

  promise = dex_promise_new ();
  g_object_set_data_full (G_OBJECT (command),
                          "DEX_PROMISE",
                          dex_ref (promise),
                          dex_unref);

  gom_adapter_queue_write (adapter,
                           plugin_devhelp_repository_delete_cb,
                           g_object_ref (command));

  return DEX_FUTURE (promise);
}

DexFuture *
plugin_devhelp_repository_find_sdk (PluginDevhelpRepository *self,
                                    const char              *ident)
{
  g_autoptr(GomFilter) filter = NULL;
  g_auto(GValue) value = G_VALUE_INIT;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self), NULL);
  g_return_val_if_fail (ident != NULL, NULL);

  g_value_init (&value, G_TYPE_STRING);
  g_value_set_string (&value, ident);

  filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_SDK, "ident", &value);

  return plugin_devhelp_repository_find_one (self, PLUGIN_TYPE_DEVHELP_SDK, filter);
}

static DexFuture *
plugin_devhelp_repository_list_fetch_cb (DexFuture *completed,
                                         gpointer   user_data)
{
  GomResourceGroup *group = user_data;
  GListStore *store = g_list_store_new (GOM_TYPE_RESOURCE);
  guint count = gom_resource_group_get_count (group);

  for (guint i = 0; i < count; i++)
    g_list_store_append (store, gom_resource_group_get_index (group, i));

  return dex_future_new_take_object (store);
}

static DexFuture *
plugin_devhelp_repository_list_find_cb (DexFuture *completed,
                                        gpointer   user_data)
{
  g_autoptr(GomResourceGroup) group = NULL;
  DexFuture *future;
  guint count;

  g_assert (DEX_IS_FUTURE (completed));

  group = dex_await_object (dex_ref (completed), NULL);
  count = gom_resource_group_get_count (group);

  future = gom_resource_group_fetch (group, 0, count);
  future = dex_future_then (future,
                            plugin_devhelp_repository_list_fetch_cb,
                            g_object_ref (group),
                            g_object_unref);

  return future;
}

DexFuture *
plugin_devhelp_repository_list (PluginDevhelpRepository *self,
                                GType                    resource_type,
                                GomFilter               *filter)
{
  DexFuture *future;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self), NULL);
  g_return_val_if_fail (g_type_is_a (resource_type, GOM_TYPE_RESOURCE), NULL);
  g_return_val_if_fail (!filter || GOM_IS_FILTER (filter), NULL);

  future = gom_repository_find (GOM_REPOSITORY (self), resource_type, filter);
  future = dex_future_then (future, plugin_devhelp_repository_list_find_cb, NULL, NULL);

  return future;
}

DexFuture *
plugin_devhelp_repository_list_sorted (PluginDevhelpRepository *self,
                                       GType                    resource_type,
                                       GomFilter               *filter,
                                       GomSorting              *sorting)
{
  DexFuture *future;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self), NULL);
  g_return_val_if_fail (g_type_is_a (resource_type, GOM_TYPE_RESOURCE), NULL);
  g_return_val_if_fail (!filter || GOM_IS_FILTER (filter), NULL);
  g_return_val_if_fail (!sorting || GOM_IS_SORTING (sorting), NULL);

  future = gom_repository_find_sorted (GOM_REPOSITORY (self), resource_type, filter, sorting);
  future = dex_future_then (future, plugin_devhelp_repository_list_find_cb, NULL, NULL);

  return future;
}

static DexFuture *
plugin_devhelp_repository_count_find_cb (DexFuture *completed,
                                         gpointer   user_data)
{
  g_autoptr(GomResourceGroup) group = NULL;
  guint count;

  g_assert (DEX_IS_FUTURE (completed));

  group = dex_await_object (dex_ref (completed), NULL);
  count = gom_resource_group_get_count (group);

  return dex_future_new_for_uint (count);
}

DexFuture *
plugin_devhelp_repository_count (PluginDevhelpRepository *self,
                                 GType                    resource_type,
                                 GomFilter               *filter)
{
  DexFuture *future;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self), NULL);
  g_return_val_if_fail (g_type_is_a (resource_type, GOM_TYPE_RESOURCE), NULL);
  g_return_val_if_fail (!filter || GOM_IS_FILTER (filter), NULL);

  future = gom_repository_find (GOM_REPOSITORY (self), resource_type, filter);
  future = dex_future_then (future,
                            plugin_devhelp_repository_count_find_cb,
                            NULL, NULL);

  return future;
}

const char *
plugin_devhelp_repository_get_cached_book_title (PluginDevhelpRepository *self,
                                                 gint64                   book_id)
{
  g_autoptr(GomResourceGroup) group = NULL;
  const char *title;

  if ((title = g_hash_table_lookup (self->cached_book_titles, &book_id)))
    return title;

  g_hash_table_remove_all (self->cached_book_titles);

  group = gom_repository_find_sync (GOM_REPOSITORY (self),
                                    PLUGIN_TYPE_DEVHELP_BOOK,
                                    NULL,
                                    NULL);

  if (group != NULL)
    {
      guint count = gom_resource_group_get_count (group);

      gom_resource_group_fetch_sync (group, 0, count, NULL);

      for (guint i = 0; i < count; i++)
        {
          PluginDevhelpBook *book = PLUGIN_DEVHELP_BOOK (gom_resource_group_get_index (group, i));
          gint64 this_id = plugin_devhelp_book_get_id (book);
          const char *this_title = plugin_devhelp_book_get_title (book);
          char *copy = g_strdup (this_title);

          if (book_id == this_id)
            title = copy;

          g_hash_table_insert (self->cached_book_titles,
                               g_memdup2 (&this_id, sizeof this_id),
                               copy);
        }
    }

  return title;
}

const char *
plugin_devhelp_repository_get_cached_sdk_title (PluginDevhelpRepository *self,
                                                gint64                   sdk_id)
{
  g_autoptr(GomResourceGroup) group = NULL;
  const char *title;

  if ((title = g_hash_table_lookup (self->cached_sdk_titles, &sdk_id)))
    return title;

  g_hash_table_remove_all (self->cached_sdk_titles);

  group = gom_repository_find_sync (GOM_REPOSITORY (self),
                                    PLUGIN_TYPE_DEVHELP_SDK,
                                    NULL,
                                    NULL);

  if (group != NULL)
    {
      guint count = gom_resource_group_get_count (group);

      gom_resource_group_fetch_sync (group, 0, count, NULL);

      for (guint i = 0; i < count; i++)
        {
          PluginDevhelpSdk *sdk = PLUGIN_DEVHELP_SDK (gom_resource_group_get_index (group, i));
          gint64 this_id = plugin_devhelp_sdk_get_id (sdk);
          char *this_title = plugin_devhelp_sdk_dup_title (sdk);

          if (sdk_id == this_id)
            title = this_title;

          g_hash_table_insert (self->cached_sdk_titles,
                               g_memdup2 (&this_id, sizeof this_id),
                               this_title);
        }
    }

  return title;
}

gint64
plugin_devhelp_repository_get_cached_sdk_id (PluginDevhelpRepository *self,
                                             gint64                   book_id)
{
  g_autoptr(GomResourceGroup) group = NULL;
  gpointer ret;
  gint64 sdk_id = 0;

  if ((ret = g_hash_table_lookup (self->cached_book_to_sdk_id, &book_id)))
    return *(gint64 *)ret;

  g_hash_table_remove_all (self->cached_book_to_sdk_id);

  group = gom_repository_find_sync (GOM_REPOSITORY (self),
                                    PLUGIN_TYPE_DEVHELP_BOOK,
                                    NULL,
                                    NULL);

  if (group != NULL)
    {
      guint count = gom_resource_group_get_count (group);

      gom_resource_group_fetch_sync (group, 0, count, NULL);

      for (guint i = 0; i < count; i++)
        {
          PluginDevhelpBook *book = PLUGIN_DEVHELP_BOOK (gom_resource_group_get_index (group, i));
          gint64 this_book_id = plugin_devhelp_book_get_id (book);
          gint64 this_sdk_id = plugin_devhelp_book_get_sdk_id (book);

          if (book_id == this_book_id)
            sdk_id = this_sdk_id;

          g_hash_table_insert (self->cached_book_to_sdk_id,
                               g_memdup2 (&this_book_id, sizeof this_book_id),
                               g_memdup2 (&this_sdk_id, sizeof this_sdk_id));
        }
    }

  return sdk_id;
}

static int
compare_version (const char *a,
                 const char *b)
{
  if (g_strcmp0 (a, "master") == 0)
    return -1;

  if (g_strcmp0 (b, "master") == 0)
    return 1;

  return g_strcmp0 (a, b);
}

static int
sort_by_name (gconstpointer a,
              gconstpointer b)
{
  PluginDevhelpSdk * const *sdk_a = a;
  PluginDevhelpSdk * const *sdk_b = b;
  const char *name_a = plugin_devhelp_sdk_get_name (*sdk_a);
  const char *name_b = plugin_devhelp_sdk_get_name (*sdk_b);

  if (strcmp (name_a, "JHBuild") == 0)
    return -1;
  else if (strcmp (name_b, "JHBuild") == 0)
    return 1;

  if (g_str_has_prefix (name_a, "GNOME ") &&
      g_str_has_prefix (name_b, "GNOME "))
    return strcmp (name_b, name_a);

  if (g_str_has_prefix (name_a, "GNOME "))
    return -1;
  else if (g_str_has_prefix (name_b, "GNOME "))
    return 1;

  return strcmp (name_a, name_b);
}

static DexFuture *
plugin_devhelp_repository_filter_sdk_by_newest (DexFuture *completed,
                                                gpointer   user_data)
{
  g_autoptr(GListModel) model = dex_await_object (dex_ref (completed), NULL);
  g_autoptr(GListStore) newest = g_list_store_new (PLUGIN_TYPE_DEVHELP_SDK);
  g_autoptr(GHashTable) hash = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_object_unref);
  g_autoptr(GPtrArray) values = NULL;
  guint n_items = g_list_model_get_n_items (model);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(PluginDevhelpSdk) sdk = g_list_model_get_item (model, i);
      const char *name = plugin_devhelp_sdk_get_name (sdk);
      PluginDevhelpSdk *prev;

      if (name == NULL)
        name = "host";

      if (!(prev = g_hash_table_lookup (hash, name)) ||
          compare_version (plugin_devhelp_sdk_get_version (sdk), plugin_devhelp_sdk_get_version (prev)) > 0)
        g_hash_table_replace (hash, g_strdup (name), g_object_ref (sdk));
    }

  values = g_hash_table_get_values_as_ptr_array (hash);
  g_ptr_array_sort (values, sort_by_name);

  g_list_store_splice (newest, 0, 0, values->pdata, values->len);

  return dex_future_new_take_object (g_steal_pointer (&newest));
}

DexFuture *
plugin_devhelp_repository_list_sdks_by_newest (PluginDevhelpRepository *self)
{
  DexFuture *future;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self), NULL);

  future = plugin_devhelp_repository_list_sdks (self);
  future = dex_future_then (future,
                            plugin_devhelp_repository_filter_sdk_by_newest,
                            g_object_ref (self),
                            g_object_unref);

  return future;
}
