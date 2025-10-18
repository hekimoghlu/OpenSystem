/* plugin-devhelp-purge-missing.c
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
#include "plugin-devhelp-purge-missing.h"

struct _PluginDevhelpPurgeMissing
{
  GObject parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginDevhelpPurgeMissing, plugin_devhelp_purge_missing, G_TYPE_OBJECT)

static DexFuture *
plugin_devhelp_purge_missing_run_fiber (gpointer data)
{
  PluginDevhelpRepository *repository = data;
  g_autoptr(GListModel) books = NULL;
  g_autoptr(GListModel) sdks = NULL;

  g_assert (PLUGIN_IS_DEVHELP_REPOSITORY (repository));

  books = dex_await_object (plugin_devhelp_repository_list (repository, PLUGIN_TYPE_DEVHELP_BOOK, NULL), NULL);

  if (books != NULL)
    {
      guint n_items = g_list_model_get_n_items (books);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(PluginDevhelpBook) book = g_list_model_get_item (books, i);
          const char *uri = plugin_devhelp_book_get_uri (book);
          g_autoptr(GFile) file = g_file_new_for_uri (uri);
          g_auto(GValue) book_id = G_VALUE_INIT;
          g_autoptr(GomFilter) book_id_filter = NULL;

          if (dex_await_boolean (dex_file_query_exists (file), NULL))
            continue;

          g_value_init (&book_id, G_TYPE_INT64);
          g_value_set_int64 (&book_id, plugin_devhelp_book_get_id (book));

          book_id_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_KEYWORD, "book-id", &book_id);
          dex_await (plugin_devhelp_repository_delete (repository,
                                                PLUGIN_TYPE_DEVHELP_KEYWORD,
                                                book_id_filter),
                     NULL);
          g_clear_object (&book_id_filter);

          book_id_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_HEADING, "book-id", &book_id);
          dex_await (plugin_devhelp_repository_delete (repository,
                                                PLUGIN_TYPE_DEVHELP_HEADING,
                                                book_id_filter),
                     NULL);
          g_clear_object (&book_id_filter);

          dex_await (gom_resource_delete (GOM_RESOURCE (book)), NULL);
        }
    }

  sdks = dex_await_object (plugin_devhelp_repository_list (repository, PLUGIN_TYPE_DEVHELP_SDK, NULL), NULL);

  if (sdks != NULL)
    {
      guint n_items = g_list_model_get_n_items (sdks);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(PluginDevhelpSdk) sdk = g_list_model_get_item (G_LIST_MODEL (sdks), i);
          g_auto(GValue) sdk_id = G_VALUE_INIT;
          g_autoptr(GomFilter) sdk_filter = NULL;
          g_autoptr(GError) error = NULL;
          guint count;

          g_value_init (&sdk_id, G_TYPE_INT64);
          g_value_set_int64 (&sdk_id, plugin_devhelp_sdk_get_id (sdk));

          sdk_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_BOOK, "sdk-id", &sdk_id);

          count = dex_await_uint (plugin_devhelp_repository_count (repository, PLUGIN_TYPE_DEVHELP_BOOK, sdk_filter), &error);

          if (error == NULL && count == 0)
            {
              g_autoptr(GomFilter) id_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_SDK, "id", &sdk_id);
              dex_await (plugin_devhelp_repository_delete (repository, PLUGIN_TYPE_DEVHELP_SDK, id_filter), NULL);
            }
        }
    }

  return dex_future_new_for_boolean (TRUE);
}

DexFuture *
plugin_devhelp_purge_missing_run (PluginDevhelpPurgeMissing *self,
                                  PluginDevhelpRepository   *repository)
{
  g_assert (PLUGIN_IS_DEVHELP_PURGE_MISSING (self));
  g_assert (PLUGIN_IS_DEVHELP_REPOSITORY (repository));

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (),
                              0,
                              plugin_devhelp_purge_missing_run_fiber,
                              g_object_ref (repository),
                              g_object_unref);
}

static void
plugin_devhelp_purge_missing_class_init (PluginDevhelpPurgeMissingClass *klass)
{
}

static void
plugin_devhelp_purge_missing_init (PluginDevhelpPurgeMissing *self)
{
}

PluginDevhelpPurgeMissing *
plugin_devhelp_purge_missing_new (void)
{
  return g_object_new (PLUGIN_TYPE_DEVHELP_PURGE_MISSING, NULL);
}
