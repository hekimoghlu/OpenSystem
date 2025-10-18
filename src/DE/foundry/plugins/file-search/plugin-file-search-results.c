/* plugin-file-search-results.c
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

#include "plugin-file-search-result.h"
#include "plugin-file-search-results.h"

#define EGG_ARRAY_ELEMENT_TYPE PluginFileSearchResult *
#define EGG_ARRAY_NAME objects
#define EGG_ARRAY_TYPE_NAME Objects
#define EGG_ARRAY_FREE_FUNC g_object_unref
#include "eggarrayimpl.c"

struct _PluginFileSearchResults
{
  GObject            parent_instance;
  GFile             *workdir;
  FoundryFuzzyIndex *index;
  GArray            *matches;
  Objects            objects;
};

static GType
plugin_file_search_results_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_SEARCH_RESULT;
}

static guint
plugin_file_search_results_get_n_items (GListModel *model)
{
  return PLUGIN_FILE_SEARCH_RESULTS (model)->matches->len;
}

static gpointer
plugin_file_search_results_get_item (GListModel *model,
                                     guint       position)
{
  PluginFileSearchResults *self = PLUGIN_FILE_SEARCH_RESULTS (model);

  if (position >= self->matches->len)
    return NULL;

  if (objects_get (&self->objects, position) == NULL)
    {
      FoundryFuzzyIndexMatch *match = &g_array_index (self->matches, FoundryFuzzyIndexMatch, position);
      *objects_index (&self->objects, position) = plugin_file_search_result_new (self->workdir, match->key, match->score);
    }

  return g_object_ref (objects_get (&self->objects, position));
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = plugin_file_search_results_get_item_type;
  iface->get_n_items = plugin_file_search_results_get_n_items;
  iface->get_item = plugin_file_search_results_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (PluginFileSearchResults, plugin_file_search_results, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static void
plugin_file_search_results_finalize (GObject *object)
{
  PluginFileSearchResults *self = (PluginFileSearchResults *)object;

  g_clear_pointer (&self->matches, g_array_unref);
  g_clear_pointer (&self->index, foundry_fuzzy_index_unref);
  objects_clear (&self->objects);
  g_clear_object (&self->workdir);

  G_OBJECT_CLASS (plugin_file_search_results_parent_class)->finalize (object);
}

static void
plugin_file_search_results_class_init (PluginFileSearchResultsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = plugin_file_search_results_finalize;
}

static void
plugin_file_search_results_init (PluginFileSearchResults *self)
{
}

PluginFileSearchResults *
plugin_file_search_results_new (GFile             *workdir,
                                FoundryFuzzyIndex *index,
                                GArray            *matches)
{
  PluginFileSearchResults *self;

  g_return_val_if_fail (index != NULL, NULL);
  g_return_val_if_fail (matches != NULL, NULL);

  self = g_object_new (PLUGIN_TYPE_FILE_SEARCH_RESULTS, NULL);
  self->workdir = g_object_ref (workdir);
  self->index = index;
  self->matches = matches;

  objects_init (&self->objects);
  objects_set_size (&self->objects, matches->len);

  return self;
}
