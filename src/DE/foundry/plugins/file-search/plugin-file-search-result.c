/* plugin-file-search-result.c
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

#include <glib/gi18n-lib.h>

#include "plugin-file-search-result.h"

struct _PluginFileSearchResult
{
  GObject parent_instance;
  GFile *workdir;
  char *filename;
  gdouble score;
};

G_DEFINE_FINAL_TYPE (PluginFileSearchResult, plugin_file_search_result, FOUNDRY_TYPE_SEARCH_RESULT)

static GIcon *
plugin_file_search_result_dup_icon (FoundrySearchResult *result)
{
  PluginFileSearchResult *self = PLUGIN_FILE_SEARCH_RESULT (result);

  return foundry_file_manager_find_symbolic_icon (NULL, NULL, self->filename);
}

static char *
plugin_file_search_result_dup_title (FoundrySearchResult *result)
{
  return g_strdup (PLUGIN_FILE_SEARCH_RESULT (result)->filename);
}

static char *
plugin_file_search_result_dup_subtitle (FoundrySearchResult *result)
{
  return g_strdup (_("Open file or folder"));
}

static DexFuture *
plugin_file_search_result_load (FoundrySearchResult *result)
{
  PluginFileSearchResult *self = PLUGIN_FILE_SEARCH_RESULT (result);

  return dex_future_new_take_object (g_file_get_child (self->workdir, self->filename));
}

static void
plugin_file_search_result_finalize (GObject *object)
{
  PluginFileSearchResult *self = (PluginFileSearchResult *)object;

  g_clear_pointer (&self->filename, g_free);
  g_clear_object (&self->workdir);

  G_OBJECT_CLASS (plugin_file_search_result_parent_class)->finalize (object);
}

static void
plugin_file_search_result_class_init (PluginFileSearchResultClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundrySearchResultClass *search_result_class = FOUNDRY_SEARCH_RESULT_CLASS (klass);

  object_class->finalize = plugin_file_search_result_finalize;

  search_result_class->dup_icon = plugin_file_search_result_dup_icon;
  search_result_class->dup_title = plugin_file_search_result_dup_title;
  search_result_class->dup_subtitle = plugin_file_search_result_dup_subtitle;
  search_result_class->load = plugin_file_search_result_load;
}

static void
plugin_file_search_result_init (PluginFileSearchResult *self)
{
}

PluginFileSearchResult *
plugin_file_search_result_new (GFile      *workdir,
                               const char *filename,
                               gdouble     score)
{
  PluginFileSearchResult *self;

  self = g_object_new (PLUGIN_TYPE_FILE_SEARCH_RESULT, NULL);
  self->workdir = g_object_ref (workdir);
  self->filename = g_strdup (filename);
  self->score = score;

  return self;
}
