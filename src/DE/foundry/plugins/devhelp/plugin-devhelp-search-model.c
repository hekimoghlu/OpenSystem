/* plugin-devhelp-search-model.c
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

#include "plugin-devhelp-navigatable.h"
#include "plugin-devhelp-search-model.h"
#include "plugin-devhelp-search-result.h"

#define PER_FETCH_GROUP 100

struct _PluginDevhelpSearchModel
{
  GObject           parent_instance;
  GomResourceGroup *group;
  GPtrArray        *prefetch;
  GHashTable       *items;
  GQueue            known;
  guint             had_prefetch : 1;
};

static void
_dex_xunref (gpointer instance)
{
  if (instance)
    dex_unref (instance);
}

static GType
plugin_devhelp_search_model_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_DOCUMENTATION;
}

static guint
plugin_devhelp_search_model_get_n_items (GListModel *model)
{
  PluginDevhelpSearchModel *self = PLUGIN_DEVHELP_SEARCH_MODEL (model);

  if (self->group != NULL)
    return gom_resource_group_get_count (self->group);

  return 0;
}

static DexFuture *
plugin_devhelp_search_model_fetch_item_cb (DexFuture *completed,
                                           gpointer   user_data)
{
  g_autoptr(GomResourceGroup) group = NULL;
  PluginDevhelpSearchResult *result = user_data;
  GomResource *resource;
  guint position;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (PLUGIN_IS_DEVHELP_SEARCH_RESULT (result));

  position = plugin_devhelp_search_result_get_position (result);

  group = dex_await_object (dex_ref (completed), NULL);
  g_assert (GOM_IS_RESOURCE_GROUP (group));
  g_assert (position < gom_resource_group_get_count (group));

  resource = gom_resource_group_get_index (group, position);
  g_assert (!resource || GOM_IS_RESOURCE (resource));

  if (resource != NULL)
    {
      g_autoptr(PluginDevhelpNavigatable) navigatable = NULL;

      navigatable = plugin_devhelp_navigatable_new_for_resource (G_OBJECT (resource));
      plugin_devhelp_search_result_set_item (result, navigatable);
    }

  return dex_future_new_for_boolean (TRUE);
}

static gpointer
plugin_devhelp_search_model_get_item (GListModel *model,
                                      guint       position)
{
  PluginDevhelpSearchModel *self = PLUGIN_DEVHELP_SEARCH_MODEL (model);
  PluginDevhelpSearchResult *result;
  DexFuture *fetch = NULL;
  guint fetch_index;

  if (self->group == NULL)
    return NULL;

  if (position >= gom_resource_group_get_count (self->group))
    return NULL;

  /* If we already got this item before, give the same pointer again */
  if ((result = g_hash_table_lookup (self->items, GUINT_TO_POINTER (position))))
    return g_object_ref (result);

  if (!self->had_prefetch)
    {
      fetch_index = position / PER_FETCH_GROUP;
      if (fetch_index >= self->prefetch->len)
        g_ptr_array_set_size (self->prefetch, fetch_index+1);

      if (!(fetch = g_ptr_array_index (self->prefetch, fetch_index)))
        {
          fetch = gom_resource_group_fetch (self->group,
                                            fetch_index * PER_FETCH_GROUP,
                                            PER_FETCH_GROUP);
          g_ptr_array_index (self->prefetch, fetch_index) = fetch;
        }
    }

  result = plugin_devhelp_search_result_new (position);
  result->model = self;
  g_queue_push_head_link (&self->known, &result->link);

  /* Make sure we have a stable item across get calls */
  g_hash_table_insert (self->items,
                       GUINT_TO_POINTER (position),
                       result);

  if (self->had_prefetch)
    {
      GomResource *resource = gom_resource_group_get_index (self->group, position);

      g_assert (fetch == NULL);

      g_assert (!resource || GOM_IS_RESOURCE (resource));

      if (resource != NULL)
        {
          g_autoptr(PluginDevhelpNavigatable) navigatable = NULL;

          navigatable = plugin_devhelp_navigatable_new_for_resource (G_OBJECT (resource));
          plugin_devhelp_search_result_set_item (result, navigatable);
        }
    }
  else
    {
      g_assert (fetch != NULL);

      dex_future_disown (dex_future_then (dex_ref (fetch),
                                          plugin_devhelp_search_model_fetch_item_cb,
                                          g_object_ref (result),
                                          g_object_unref));
    }

  return result;
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_n_items = plugin_devhelp_search_model_get_n_items;
  iface->get_item_type = plugin_devhelp_search_model_get_item_type;
  iface->get_item = plugin_devhelp_search_model_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (PluginDevhelpSearchModel, plugin_devhelp_search_model, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

enum {
  PROP_0,
  PROP_GROUP,
  N_PROPS
};

static GParamSpec *properties [N_PROPS];

PluginDevhelpSearchModel *
plugin_devhelp_search_model_new (GomResourceGroup *group,
                                 gboolean          had_prefetch)
{
  PluginDevhelpSearchModel *self;

  g_return_val_if_fail (GOM_IS_RESOURCE_GROUP (group), NULL);

  self = g_object_new (PLUGIN_TYPE_DEVHELP_SEARCH_MODEL,
                       "group", group,
                       NULL);
  self->had_prefetch = !!had_prefetch;

  return self;
}

static void
plugin_devhelp_search_model_dispose (GObject *object)
{
  PluginDevhelpSearchModel *self = (PluginDevhelpSearchModel *)object;
  GList *iter;

  while ((iter = self->known.head))
    {
      PluginDevhelpSearchResult *result = iter->data;
      plugin_devhelp_search_model_release (self, result);
    }

  g_clear_pointer (&self->prefetch, g_ptr_array_unref);
  g_clear_pointer (&self->items, g_hash_table_unref);
  g_clear_object (&self->group);

  G_OBJECT_CLASS (plugin_devhelp_search_model_parent_class)->dispose (object);
}

static void
plugin_devhelp_search_model_get_property (GObject    *object,
                                          guint       prop_id,
                                          GValue     *value,
                                          GParamSpec *pspec)
{
  PluginDevhelpSearchModel *self = PLUGIN_DEVHELP_SEARCH_MODEL (object);

  switch (prop_id)
    {
    case PROP_GROUP:
      g_value_set_object (value, self->group);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_devhelp_search_model_set_property (GObject      *object,
                                          guint         prop_id,
                                          const GValue *value,
                                          GParamSpec   *pspec)
{
  PluginDevhelpSearchModel *self = PLUGIN_DEVHELP_SEARCH_MODEL (object);

  switch (prop_id)
    {
    case PROP_GROUP:
      self->group = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_devhelp_search_model_class_init (PluginDevhelpSearchModelClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = plugin_devhelp_search_model_dispose;
  object_class->get_property = plugin_devhelp_search_model_get_property;
  object_class->set_property = plugin_devhelp_search_model_set_property;

  properties[PROP_GROUP] =
    g_param_spec_object ("group", NULL, NULL,
                         GOM_TYPE_RESOURCE_GROUP,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_devhelp_search_model_init (PluginDevhelpSearchModel *self)
{
  self->prefetch = g_ptr_array_new_with_free_func (_dex_xunref);
  self->items = g_hash_table_new (NULL, NULL);
}

DexFuture *
plugin_devhelp_search_model_prefetch (PluginDevhelpSearchModel *self,
                                      guint                     position)
{
  g_autoptr(PluginDevhelpSearchResult) result = NULL;
  DexFuture *prefetch;
  guint fetch_index;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_SEARCH_MODEL (self), NULL);

  if (self->had_prefetch)
    return dex_future_new_true ();

  if (!(result = g_list_model_get_item (G_LIST_MODEL (self), position)))
    return dex_future_new_true ();

  fetch_index = position / PER_FETCH_GROUP;

  g_assert (fetch_index < self->prefetch->len);

  prefetch = g_ptr_array_index (self->prefetch, fetch_index);

  return dex_ref (prefetch);
}

void
plugin_devhelp_search_model_release (PluginDevhelpSearchModel  *self,
                                     PluginDevhelpSearchResult *result)
{
  g_return_if_fail (PLUGIN_IS_DEVHELP_SEARCH_MODEL (self));
  g_return_if_fail (PLUGIN_IS_DEVHELP_SEARCH_RESULT (result));

  if (self->items)
    g_hash_table_remove (self->items,
                         GUINT_TO_POINTER (result->position));

  g_queue_unlink (&self->known, &result->link);

  result->model = NULL;
}
