/* plugin-devhelp-documentation-provider.c
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

#include <foundry.h>

#include "foundry-gom-private.h"

#include "plugin-devhelp-book.h"
#include "plugin-devhelp-documentation-provider.h"
#include "plugin-devhelp-heading.h"
#include "plugin-devhelp-importer.h"
#include "plugin-devhelp-keyword.h"
#include "plugin-devhelp-navigatable.h"
#include "plugin-devhelp-purge-missing.h"
#include "plugin-devhelp-repository.h"
#include "plugin-devhelp-sdk.h"
#include "plugin-devhelp-search-model.h"

struct _PluginDevhelpDocumentationProvider
{
  FoundryDocumentationProvider  parent_instance;
  PluginDevhelpRepository      *repository;
};

G_DEFINE_FINAL_TYPE (PluginDevhelpDocumentationProvider, plugin_devhelp_documentation_provider, FOUNDRY_TYPE_DOCUMENTATION_PROVIDER)

static DexFuture *
plugin_devhelp_documentation_provider_load_fiber (gpointer user_data)
{
  PluginDevhelpDocumentationProvider *self = user_data;
  g_autoptr(GError) error = NULL;
  g_autofree char *dir = NULL;
  g_autofree char *path = NULL;

  g_assert (PLUGIN_IS_DEVHELP_DOCUMENTATION_PROVIDER (self));

  dir = g_build_filename (g_get_user_data_dir (), "libfoundry", "doc", NULL);
  path = g_build_filename (dir, "devhelp.sqlite", NULL);

  if (!dex_await (dex_mkdir_with_parents (dir, 0750), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(self->repository = dex_await_object (plugin_devhelp_repository_open (path), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static DexFuture *
plugin_devhelp_documentation_provider_load (FoundryDocumentationProvider *provider)
{
  g_assert (PLUGIN_IS_DEVHELP_DOCUMENTATION_PROVIDER (provider));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_devhelp_documentation_provider_load_fiber,
                              g_object_ref (provider),
                              g_object_unref);
}

static DexFuture *
plugin_devhelp_documentation_provider_unload (FoundryDocumentationProvider *provider)
{
  PluginDevhelpDocumentationProvider *self = (PluginDevhelpDocumentationProvider *)provider;

  g_assert (PLUGIN_IS_DEVHELP_DOCUMENTATION_PROVIDER (self));

  g_clear_object (&self->repository);

  return dex_future_new_true ();
}

static DexFuture *
plugin_devhelp_documentation_provider_index_fiber (PluginDevhelpDocumentationProvider *self,
                                                   GListModel                         *roots,
                                                   PluginDevhelpRepository            *repository)
{
  g_autoptr(PluginDevhelpPurgeMissing) purge_missing = NULL;
  g_autoptr(GError) error = NULL;
  guint n_items;

  g_assert (PLUGIN_IS_DEVHELP_DOCUMENTATION_PROVIDER (self));
  g_assert (G_IS_LIST_MODEL (roots));
  g_assert (PLUGIN_IS_DEVHELP_REPOSITORY (repository));

  n_items = g_list_model_get_n_items (roots);

  if (n_items > 0)
    {
      g_autoptr(PluginDevhelpImporter) importer = plugin_devhelp_importer_new ();
      g_autoptr(PluginDevhelpProgress) progress = plugin_devhelp_progress_new ();

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryDocumentationRoot) root = g_list_model_get_item (roots, i);
          g_autoptr(PluginDevhelpSdk) sdk = NULL;
          g_autoptr(GListModel) directories = foundry_documentation_root_list_directories (root);
          g_autofree char *ident = foundry_documentation_root_dup_identifier (root);
          g_autofree char *title = foundry_documentation_root_dup_title (root);
          g_autofree char *version = foundry_documentation_root_dup_version (root);
          g_autoptr(GIcon) icon = foundry_documentation_root_dup_icon (root);
          const char *icon_name = NULL;
          guint n_dirs = g_list_model_get_n_items (directories);
          gint64 sdk_id = 0;

          if (G_IS_THEMED_ICON (icon))
            icon_name = g_themed_icon_get_names (G_THEMED_ICON (icon))[0];

          /* Insert the SDK if it is not yet available */
          if (!(sdk = dex_await_object (plugin_devhelp_repository_find_sdk (repository, ident), NULL)))
            {
              sdk = g_object_new (PLUGIN_TYPE_DEVHELP_SDK,
                                  "repository", repository,
                                  "name", title,
                                  "version", version,
                                  "ident", ident,
                                  "icon-name", icon_name,
                                  NULL);

              if (!dex_await (gom_resource_save (GOM_RESOURCE (sdk)), &error))
                return dex_future_new_for_error (g_steal_pointer (&error));
            }

          sdk_id = plugin_devhelp_sdk_get_id (sdk);

          for (guint j = 0; j < n_dirs; j++)
            {
              g_autoptr(GFile) dir = g_list_model_get_item (directories, j);
              g_autofree char *path = g_file_get_path (dir);

              plugin_devhelp_importer_add_directory (importer, path, sdk_id);
            }
        }

      if (!dex_await (plugin_devhelp_importer_import (importer, repository, progress), &error))
        return dex_future_new_for_error (g_steal_pointer (&error));
    }

  /* Now purge any empty SDK entries */
  purge_missing = plugin_devhelp_purge_missing_new ();
  if (!dex_await (plugin_devhelp_purge_missing_run (purge_missing, repository), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static DexFuture *
plugin_devhelp_documentation_provider_index (FoundryDocumentationProvider *provider,
                                             GListModel                   *roots)
{
  PluginDevhelpDocumentationProvider *self = (PluginDevhelpDocumentationProvider *)provider;
  guint n_roots;

  dex_return_error_if_fail (PLUGIN_IS_DEVHELP_DOCUMENTATION_PROVIDER (self));
  dex_return_error_if_fail (G_IS_LIST_MODEL (roots));
  dex_return_error_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self->repository));

  n_roots = g_list_model_get_n_items (roots);

  g_debug ("Re-indexing documentation in %u roots", n_roots);

  for (guint i = 0; i < n_roots; i++)
    {
      g_autoptr(FoundryDocumentationRoot) root = g_list_model_get_item (roots, i);
      g_autofree char *title = foundry_documentation_root_dup_title (root);

      g_debug ("Root[%u] is \"%s\"", i, title);
    }

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_devhelp_documentation_provider_index_fiber),
                                  3,
                                  PLUGIN_TYPE_DEVHELP_DOCUMENTATION_PROVIDER, provider,
                                  G_TYPE_LIST_MODEL, roots,
                                  PLUGIN_TYPE_DEVHELP_REPOSITORY, self->repository);
}

static char *
like_string (const char *str)
{
  GString *gstr;

  if (str == NULL || str[0] == 0)
    return g_strdup ("%");

  gstr = g_string_new (NULL);
  g_string_append_c (gstr, '%');

  if (str != NULL)
    {
      g_string_append (gstr, str);
      g_string_append_c (gstr, '%');
      g_string_replace (gstr, " ", "%", 0);
    }

  return g_string_free (gstr, FALSE);
}

static DexFuture *
plugin_devhelp_documentation_provider_query_fiber (PluginDevhelpDocumentationProvider *self,
                                                   FoundryDocumentationQuery          *query,
                                                   FoundryDocumentationMatches        *matches,
                                                   PluginDevhelpRepository            *repository)
{
  g_autoptr(GListModel) sdks = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GomFilter) keyword_filter = NULL;
  g_autoptr(GomFilter) filter = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GPtrArray) prefetch = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *keyword = NULL;
  g_autofree char *function_name = NULL;
  g_autofree char *property_name = NULL;
  g_autofree char *type_name = NULL;
  gboolean prefetch_all;
  guint n_sdks;

  g_assert (PLUGIN_IS_DEVHELP_DOCUMENTATION_PROVIDER (self));
  g_assert (FOUNDRY_IS_DOCUMENTATION_QUERY (query));
  g_assert (FOUNDRY_IS_DOCUMENTATION_MATCHES (matches));
  g_assert (PLUGIN_IS_DEVHELP_REPOSITORY (repository));

  prefetch_all = foundry_documentation_query_get_prefetch_all (query);

  if ((keyword = foundry_documentation_query_dup_keyword (query)))
    {
      g_auto(GValue) like_value = G_VALUE_INIT;

      g_value_init (&like_value, G_TYPE_STRING);
      g_value_take_string (&like_value, like_string (keyword));
      keyword_filter = gom_filter_new_like (PLUGIN_TYPE_DEVHELP_KEYWORD, "name", &like_value);
    }

  function_name = foundry_documentation_query_dup_function_name (query);
  property_name = foundry_documentation_query_dup_property_name (query);
  type_name = foundry_documentation_query_dup_type_name (query);

  if (property_name && type_name)
    {
      g_auto(GValue) like_value = G_VALUE_INIT;
      g_autofree char *str = g_strdup_printf ("The %s:%s property", type_name, property_name);

      g_value_init (&like_value, G_TYPE_STRING);
      g_value_take_string (&like_value, g_steal_pointer (&str));

      filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_KEYWORD, "name", &like_value);
    }
  else if (property_name)
    {
      g_auto(GValue) like_value = G_VALUE_INIT;
      g_autofree char *str = g_strdup_printf ("The %%:%s property", property_name);

      g_value_init (&like_value, G_TYPE_STRING);
      g_value_take_string (&like_value, g_steal_pointer (&str));

      filter = gom_filter_new_like (PLUGIN_TYPE_DEVHELP_KEYWORD, "name", &like_value);
    }
  else if (function_name)
    {
      g_auto(GValue) name_value = G_VALUE_INIT;
      g_auto(GValue) kind_value = G_VALUE_INIT;
      g_autoptr(GomFilter) name_filter = NULL;
      g_autoptr(GomFilter) kind_filter = NULL;

      g_value_init (&name_value, G_TYPE_STRING);
      g_value_set_string (&name_value, function_name);
      name_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_KEYWORD, "name", &name_value);

      g_value_init (&kind_value, G_TYPE_STRING);
      g_value_set_string (&kind_value, "function");
      kind_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_KEYWORD, "kind", &kind_value);

      filter = gom_filter_new_and (name_filter, kind_filter);
    }
  else if (type_name)
    {
      g_auto(GValue) name_value = G_VALUE_INIT;
      g_auto(GValue) kind_value = G_VALUE_INIT;
      g_autoptr(GomFilter) name_filter = NULL;
      g_autoptr(GomFilter) kind_filter = NULL;

      g_value_init (&name_value, G_TYPE_STRING);
      g_value_set_string (&name_value, type_name);
      name_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_KEYWORD, "name", &name_value);

      g_value_init (&kind_value, G_TYPE_STRING);
      g_value_set_string (&kind_value, "struct");
      kind_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_KEYWORD, "kind", &kind_value);

#if 0
      /* We could have other types here, like enum, etc */
      g_value_init (&kind_value, G_TYPE_STRING);
      g_value_set_string (&kind_value, "struct");
      kind_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_KEYWORD, "kind", &kind_value);

      filter = gom_filter_new_and (name_filter, kind_filter);
#else
      filter = g_object_ref (name_filter);
#endif
    }

  if (!(sdks = dex_await_object (plugin_devhelp_repository_list_sdks_by_newest (repository), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  futures = g_ptr_array_new_with_free_func (dex_unref);
  n_sdks = g_list_model_get_n_items (sdks);

  for (guint i = 0; i < n_sdks; i++)
    {
      g_autoptr(PluginDevhelpSdk) sdk = g_list_model_get_item (sdks, i);
      g_autoptr(GArray) values = g_array_new (FALSE, TRUE, sizeof (GValue));
      g_autoptr(GString) str = g_string_new ("\"book-id\" IN (");
      g_autoptr(GListModel) books = NULL;
      g_autoptr(GomFilter) book_filter = NULL;
      g_autoptr(GomFilter) full_filter = NULL;
      guint n_books;

      if (!(books = dex_await_object (plugin_devhelp_sdk_list_books (sdk), NULL)) ||
          (n_books = g_list_model_get_n_items (books)) == 0)
        continue;

      for (guint j = 0; j < n_books; j++)
        {
          g_autoptr(PluginDevhelpBook) book = g_list_model_get_item (books, j);
          GValue value = G_VALUE_INIT;

          g_value_init (&value, G_TYPE_INT64);
          g_value_set_int64 (&value, plugin_devhelp_book_get_id (book));

          g_array_append_val (values, value);
          g_string_append_c (str, '?');

          if (j + 1 < n_books)
            g_string_append_c (str, ',');
        }

      g_string_append_c (str, ')');

      book_filter = gom_filter_new_sql (str->str, values);

      if (filter != NULL)
        full_filter = gom_filter_new_and (book_filter, filter);
      else if (keyword_filter != NULL)
        full_filter = gom_filter_new_and (book_filter, keyword_filter);
      else
        full_filter = g_object_ref (book_filter);

      g_ptr_array_add (futures,
                       gom_repository_find (GOM_REPOSITORY (repository),
                                            PLUGIN_TYPE_DEVHELP_KEYWORD,
                                            full_filter));
    }

  if (!dex_await (dex_future_allv ((DexFuture **)futures->pdata, futures->len), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  store = g_list_store_new (G_TYPE_LIST_MODEL);
  prefetch = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < futures->len; i++)
    {
      DexFuture *future = g_ptr_array_index (futures, i);
      GomResourceGroup *group = g_value_get_object (dex_future_get_value (future, NULL));

      if (prefetch_all)
        g_ptr_array_add (prefetch, gom_resource_group_fetch_all (group));
    }

  if (prefetch->len > 0)
    dex_await (foundry_future_all (prefetch), NULL);

  for (guint i = 0; i < futures->len; i++)
    {
      DexFuture *future = g_ptr_array_index (futures, i);
      GomResourceGroup *group = g_value_get_object (dex_future_get_value (future, NULL));
      g_autoptr(PluginDevhelpSearchModel) wrapped = plugin_devhelp_search_model_new (group, prefetch_all);

      foundry_documentation_matches_add_section (matches, G_LIST_MODEL (wrapped));

      if (i == 0 && g_list_model_get_n_items (G_LIST_MODEL (wrapped)) > 0)
        /* If there are any items, then wait for the first page to fetch so that
         * UI can rely on results having non-null items at early positions.
         */
        g_ptr_array_add (prefetch, plugin_devhelp_search_model_prefetch (wrapped, 0));
    }

  return dex_future_new_true ();
}

static DexFuture *
plugin_devhelp_documentation_provider_query (FoundryDocumentationProvider *provider,
                                             FoundryDocumentationQuery    *query,
                                             FoundryDocumentationMatches  *matches)
{
  PluginDevhelpDocumentationProvider *self = (PluginDevhelpDocumentationProvider *)provider;

  dex_return_error_if_fail (PLUGIN_IS_DEVHELP_DOCUMENTATION_PROVIDER (self));
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_QUERY (query));
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_MATCHES (matches));
  dex_return_error_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self->repository));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_devhelp_documentation_provider_query_fiber),
                                  4,
                                  PLUGIN_TYPE_DEVHELP_DOCUMENTATION_PROVIDER, provider,
                                  FOUNDRY_TYPE_DOCUMENTATION_QUERY, query,
                                  FOUNDRY_TYPE_DOCUMENTATION_MATCHES, matches,
                                  PLUGIN_TYPE_DEVHELP_REPOSITORY, self->repository);
}

static gpointer
sdk_to_navigatable (gpointer item,
                    gpointer user_data)
{
  g_autoptr(PluginDevhelpSdk) sdk = item;

  g_assert (PLUGIN_IS_DEVHELP_SDK (sdk));

  return plugin_devhelp_navigatable_new_for_resource (G_OBJECT (sdk));
}

static DexFuture *
plugin_devhelp_documentation_provider_list_children_fiber (FoundryDocumentationProvider *provider,
                                                           FoundryDocumentation         *parent)
{
  PluginDevhelpDocumentationProvider *self = (PluginDevhelpDocumentationProvider *)provider;

  g_assert (PLUGIN_IS_DEVHELP_DOCUMENTATION_PROVIDER (self));
  g_assert (!parent || FOUNDRY_IS_DOCUMENTATION (parent));

  if (parent == NULL)
    {
      g_autoptr(GListModel) sdks = NULL;
      g_autoptr(GError) error = NULL;

      if (!(sdks = dex_await_object (plugin_devhelp_repository_list_sdks_by_newest (self->repository), &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));

      return dex_future_new_take_object (foundry_map_list_model_new (g_object_ref (sdks),
                                                                     sdk_to_navigatable,
                                                                     NULL,
                                                                     NULL));
    }

  if (!PLUGIN_IS_DEVHELP_NAVIGATABLE (parent))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_SUPPORTED,
                                  "Not supported");

  return foundry_documentation_find_children (parent);
}

static DexFuture *
plugin_devhelp_documentation_provider_list_children (FoundryDocumentationProvider *provider,
                                                     FoundryDocumentation         *parent)
{
  PluginDevhelpDocumentationProvider *self = (PluginDevhelpDocumentationProvider *)provider;

  dex_return_error_if_fail (PLUGIN_IS_DEVHELP_DOCUMENTATION_PROVIDER (self));
  dex_return_error_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self->repository));
  dex_return_error_if_fail (!parent || FOUNDRY_IS_DOCUMENTATION (parent));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_devhelp_documentation_provider_list_children_fiber),
                                  2,
                                  PLUGIN_TYPE_DEVHELP_DOCUMENTATION_PROVIDER, provider,
                                  FOUNDRY_TYPE_DOCUMENTATION, parent);
}

static DexFuture *
plugin_devhelp_documentation_provider_find_by_uri_fiber (PluginDevhelpDocumentationProvider *self,
                                                         const char                         *uri)
{
  g_autoptr(GomResource) resource = NULL;

  g_assert (PLUGIN_IS_DEVHELP_DOCUMENTATION_PROVIDER (self));
  g_assert (PLUGIN_IS_DEVHELP_REPOSITORY (self->repository));
  g_assert (uri != NULL);

  if ((resource = dex_await_object (plugin_devhelp_heading_find_by_uri (self->repository, uri), NULL)) ||
      (resource = dex_await_object (plugin_devhelp_keyword_find_by_uri (self->repository, uri), NULL)))
    return dex_future_new_take_object (plugin_devhelp_navigatable_new_for_resource (G_OBJECT (resource)));

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}

static DexFuture *
plugin_devhelp_documentation_provider_find_by_uri (FoundryDocumentationProvider *provider,
                                                   const char                   *uri)
{
  PluginDevhelpDocumentationProvider *self = (PluginDevhelpDocumentationProvider *)provider;

  dex_return_error_if_fail (PLUGIN_IS_DEVHELP_DOCUMENTATION_PROVIDER (self));
  dex_return_error_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (self->repository));
  dex_return_error_if_fail (uri != NULL);

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_devhelp_documentation_provider_find_by_uri_fiber),
                                  2,
                                  PLUGIN_TYPE_DEVHELP_DOCUMENTATION_PROVIDER, self,
                                  G_TYPE_STRING, uri);
}

static void
plugin_devhelp_documentation_provider_class_init (PluginDevhelpDocumentationProviderClass *klass)
{
  FoundryDocumentationProviderClass *documentation_provider_class = FOUNDRY_DOCUMENTATION_PROVIDER_CLASS (klass);

  documentation_provider_class->load = plugin_devhelp_documentation_provider_load;
  documentation_provider_class->unload = plugin_devhelp_documentation_provider_unload;
  documentation_provider_class->index = plugin_devhelp_documentation_provider_index;
  documentation_provider_class->query = plugin_devhelp_documentation_provider_query;
  documentation_provider_class->list_children = plugin_devhelp_documentation_provider_list_children;
  documentation_provider_class->find_by_uri = plugin_devhelp_documentation_provider_find_by_uri;
}

static void
plugin_devhelp_documentation_provider_init (PluginDevhelpDocumentationProvider *self)
{
}
