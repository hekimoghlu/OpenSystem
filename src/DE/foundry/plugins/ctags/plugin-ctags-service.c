/* plugin-ctags-service.c
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

#include "plugin-ctags-builder.h"
#include "plugin-ctags-file.h"
#include "plugin-ctags-service.h"
#include "plugin-ctags-util.h"

struct _PluginCtagsService
{
  FoundryService parent_instance;

  /* A GListModel of PluginCtagsFile which we use to perform queries
   * against the ctags indexes.
   */
  GListStore *files;

  /* A future that is a DexFiber doing the mining of new ctags data.
   * It looks through the project to find directories which have newer
   * data than their respective tags file.
   *
   * If found, it generates the tags for that directory.
   */
  DexFuture *miner;
};

G_DEFINE_FINAL_TYPE (PluginCtagsService, plugin_ctags_service, FOUNDRY_TYPE_SERVICE)

typedef struct _DirectoryPair
{
  GFile *source_dir;
  GFile *tags_dir;
} DirectoryPair;

static void
directory_pair_clear (gpointer data)
{
  DirectoryPair *pair = data;

  g_clear_object (&pair->source_dir);
  g_clear_object (&pair->tags_dir);
}

static void
mine_directories (const char *ctags,
                  GArray     *directories)
{
  g_assert (ctags != NULL && ctags[0] != 0);
  g_assert (directories != NULL);

  while (directories->len > 0)
    {
      g_autoptr(PluginCtagsBuilder) builder = NULL;
      g_autoptr(GFileEnumerator) enumerator = NULL;
      g_autoptr(GDateTime) most_recent_change = NULL;
      g_autoptr(GError) error = NULL;
      g_autoptr(GFile) source_dir = NULL;
      g_autoptr(GFile) tags_dir = NULL;

      source_dir = g_array_index (directories, DirectoryPair, directories->len-1).source_dir;
      tags_dir = g_array_index (directories, DirectoryPair, directories->len-1).tags_dir;
      directories->len--;

      builder = plugin_ctags_builder_new (tags_dir);
      plugin_ctags_builder_set_ctags_path (builder, ctags);

#if 0
      g_print ("Querying %s => `%s/tags`\n",
               g_file_peek_path (source_dir),
               g_file_peek_path (tags_dir));
#endif

      if (!(enumerator = dex_await_object (dex_file_enumerate_children (source_dir,
                                                                        (G_FILE_ATTRIBUTE_STANDARD_NAME","
                                                                         G_FILE_ATTRIBUTE_STANDARD_TYPE","
                                                                         G_FILE_ATTRIBUTE_TIME_MODIFIED","),
                                                                        G_FILE_QUERY_INFO_NOFOLLOW_SYMLINKS,
                                                                        G_PRIORITY_DEFAULT),
                                           &error)))
        goto handle_error;

      for (;;)
        {
          g_autolist(GFileInfo) infos = dex_await_boxed (dex_file_enumerator_next_files (enumerator, 100, 0), &error);

          if (error != NULL)
            goto handle_error;

          if (infos == NULL)
            break;

          for (const GList *iter = infos; iter; iter = iter->next)
            {
              GFileInfo *info = iter->data;
              GFileType file_type = g_file_info_get_file_type (info);
              const char *name = g_file_info_get_name (info);

              if (file_type != G_FILE_TYPE_REGULAR)
                {
                  if (file_type == G_FILE_TYPE_DIRECTORY && name[0] != '.')
                    {
                      DirectoryPair pair;

                      pair.source_dir = g_file_enumerator_get_child (enumerator, info);
                      pair.tags_dir = g_file_get_child (tags_dir, name);
                      g_array_append_val (directories, pair);
                    }
                }
              else if (plugin_ctags_is_indexable (name))
                {
                  g_autoptr(GFile) file = g_file_enumerator_get_child (enumerator, info);
                  g_autoptr(GDateTime) when = g_file_info_get_modification_date_time (info);

                  plugin_ctags_builder_add_file (builder, file);

                  if (most_recent_change == NULL ||
                      g_date_time_compare (when, most_recent_change) > 0)
                    most_recent_change = g_steal_pointer (&when);
                }
            }
        }

    handle_error:
      if (g_error_matches (error, DEX_ERROR, DEX_ERROR_FIBER_CANCELLED))
        return;
    }
}

static DexFuture *
plugin_ctags_service_miner_fiber (gpointer data)
{
  PluginCtagsService *self = data;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GSettings) settings = NULL;
  g_autoptr(GArray) directories = NULL;
  g_autoptr(GFile) workdir = NULL;
  g_autoptr(GFile) tagsdir = NULL;
  g_autofree char *ctags = NULL;
  DirectoryPair root;

  g_assert (PLUGIN_IS_CTAGS_SERVICE (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  workdir = foundry_context_dup_project_directory (context);
  tagsdir = foundry_context_cache_file (context, "ctags", NULL);
  settings = g_settings_new ("app.devsuite.foundry.ctags");
  ctags = g_settings_get_string (settings, "path");

  if (foundry_str_empty0 (ctags))
    g_set_str (&ctags, "ctags");

  directories = g_array_new (FALSE, FALSE, sizeof (DirectoryPair));
  g_array_set_clear_func (directories, directory_pair_clear);
  root.source_dir = g_steal_pointer (&workdir);
  root.tags_dir = g_steal_pointer (&tagsdir);
  g_array_append_val (directories, root);

  mine_directories (ctags, directories);

  return dex_future_new_true ();
}

static DexFuture *
plugin_ctags_service_start_fiber (gpointer data)
{
  PluginCtagsService *self = data;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) files = NULL;
  g_autoptr(GFile) ctags_file = NULL;
  g_autofree char *ctags_dir = NULL;

  g_assert (PLUGIN_IS_CTAGS_SERVICE (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  ctags_dir = foundry_context_cache_filename (context, "ctags", NULL);
  ctags_file = g_file_new_for_path (ctags_dir);

  if ((files = dex_await_boxed (foundry_file_find_with_depth (ctags_file, "tags", 10), NULL)))
    {
      g_autoptr(GPtrArray) futures = g_ptr_array_new_with_free_func (dex_unref);

      for (guint i = 0; i < files->len; i++)
        {
          GFile *file = g_ptr_array_index (files, i);

          g_ptr_array_add (futures, plugin_ctags_file_new (file));
        }

      if (futures->len > 0)
        dex_await (foundry_future_all (futures), NULL);

      for (guint i = 0; i < futures->len; i++)
        {
          DexFuture *future = g_ptr_array_index (futures, i);
          const GValue *value;

          if ((value = dex_future_get_value (future, NULL)))
            g_list_store_append (self->files, g_value_get_object (value));
        }
    }

  return dex_future_new_true ();
}

static DexFuture *
plugin_ctags_service_start (FoundryService *service)
{
  g_assert (PLUGIN_IS_CTAGS_SERVICE (service));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_ctags_service_start_fiber,
                              g_object_ref (service),
                              g_object_unref);
}

static DexFuture *
plugin_ctags_service_stop (FoundryService *service)
{
  PluginCtagsService *self = (PluginCtagsService *)service;

  g_assert (PLUGIN_IS_CTAGS_SERVICE (self));

  g_list_store_remove_all (self->files);
  dex_clear (&self->miner);

  return dex_future_new_true ();
}

static void
plugin_ctags_service_finalize (GObject *object)
{
  PluginCtagsService *self = (PluginCtagsService *)object;

  dex_clear (&self->miner);
  g_clear_object (&self->files);

  G_OBJECT_CLASS (plugin_ctags_service_parent_class)->finalize (object);
}

static void
plugin_ctags_service_class_init (PluginCtagsServiceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->finalize = plugin_ctags_service_finalize;

  service_class->start = plugin_ctags_service_start;
  service_class->stop = plugin_ctags_service_stop;
}

static void
plugin_ctags_service_init (PluginCtagsService *self)
{
  self->files = g_list_store_new (PLUGIN_TYPE_CTAGS_FILE);
}

static void
plugin_ctags_service_ensure_mined (PluginCtagsService *self)
{
  g_assert (PLUGIN_IS_CTAGS_SERVICE (self));

  if (self->miner != NULL)
    return;

  /* Now start the miner. We do not block startup on this because we
   * wouldn't want it to prevent shutdown of the service. So we create
   * a new fiber for the miner which may be discarded in stop(), and
   * thusly, potentially cancel the fiber.
   */
  self->miner = dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                                     plugin_ctags_service_miner_fiber,
                                     g_object_ref (self),
                                     g_object_unref);
}

/**
 * plugin_ctags_service_list_files:
 *
 * Returns: (transfer full):
 */
GListModel *
plugin_ctags_service_list_files (PluginCtagsService *self)
{
  g_return_val_if_fail (PLUGIN_IS_CTAGS_SERVICE (self), NULL);

  plugin_ctags_service_ensure_mined (self);

  return g_object_ref (G_LIST_MODEL (self->files));
}
