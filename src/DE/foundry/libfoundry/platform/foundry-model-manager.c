/* foundry-model-manager.c
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

#include "eggflattenlistmodel.h"
#include "eggmaplistmodel.h"

#include "foundry-debug.h"
#include "foundry-model-manager.h"

G_DEFINE_TYPE (FoundryModelManager, foundry_model_manager, G_TYPE_OBJECT)

static GListModel *
foundry_model_manager_real_flatten (FoundryModelManager *self,
                                    GListModel          *model)
{
  return G_LIST_MODEL (egg_flatten_list_model_new (model));
}

static GListModel *
foundry_model_manager_real_map (FoundryModelManager     *self,
                                GListModel              *model,
                                FoundryListModelMapFunc  map_func,
                                gpointer                 user_data,
                                GDestroyNotify           user_destroy)
{
  return G_LIST_MODEL (egg_map_list_model_new (model, map_func, user_data, user_destroy));
}

static void
foundry_model_manager_class_init (FoundryModelManagerClass *klass)
{
  klass->flatten = foundry_model_manager_real_flatten;
  klass->map = foundry_model_manager_real_map;
}

static void
foundry_model_manager_init (FoundryModelManager *self)
{
}

/**
 * foundry_model_manager_flatten:
 * @self: a [class@Foundry.ModelManager]
 * @model: (transfer full) (nullable): a [iface@Gio.ListModel]
 *
 * Returns: (transfer full):
 */
GListModel *
foundry_model_manager_flatten (FoundryModelManager *self,
                               GListModel          *model)
{
  return FOUNDRY_MODEL_MANAGER_GET_CLASS (self)->flatten (self, model);
}

/**
 * foundry_model_manager_map:
 * @self: a [class@Foundry.ModelManager]
 * @model: (transfer full) (nullable): a [iface@Gio.ListModel]
 *
 * Returns: (transfer full):
 */
GListModel *
foundry_model_manager_map (FoundryModelManager     *self,
                           GListModel              *model,
                           FoundryListModelMapFunc  map_func,
                           gpointer                 user_data,
                           GDestroyNotify           user_destroy)
{
  return FOUNDRY_MODEL_MANAGER_GET_CLASS (self)->map (self, model, map_func, user_data, user_destroy);
}

static G_LOCK_DEFINE (default_instance);
static FoundryModelManager *default_instance;

/**
 * foundry_model_manager_dup_default:
 *
 * Returns: (transfer full):
 */
FoundryModelManager *
foundry_model_manager_dup_default (void)
{
  FoundryModelManager *ret;

  G_LOCK (default_instance);
  if (default_instance == NULL)
    default_instance = g_object_new (FOUNDRY_TYPE_MODEL_MANAGER, NULL);
  ret = g_object_ref (default_instance);
  G_UNLOCK (default_instance);

  return ret;
}

void
foundry_model_manager_set_default (FoundryModelManager *self)
{
  g_return_if_fail (!self || FOUNDRY_IS_MODEL_MANAGER (self));

  G_LOCK (default_instance);
  g_set_object (&default_instance, self);
  G_UNLOCK (default_instance);
}

/**
 * foundry_flatten_list_model_new:
 * @model: (transfer full) (nullable):
 *
 * Returns: (transfer full):
 */
GListModel *
foundry_flatten_list_model_new (GListModel *model)
{
  g_autoptr(FoundryModelManager) self = foundry_model_manager_dup_default ();

  return foundry_model_manager_flatten (self, model);
}

/**
 * foundry_map_list_model_new:
 * @model: (transfer full) (nullable): a [iface@Gio.ListModel]
 *
 * Returns: (transfer full):
 */
GListModel *
foundry_map_list_model_new (GListModel              *model,
                            FoundryListModelMapFunc  map_func,
                            gpointer                 user_data,
                            GDestroyNotify           user_destroy)
{
  g_autoptr(FoundryModelManager) self = foundry_model_manager_dup_default ();

  return foundry_model_manager_map (self, model, map_func, user_destroy, user_destroy);
}

/**
 * foundry_list_model_set_future:
 * @future: (nullable): a [class@Dex.Future] or %NULL
 *
 * Sets the future that can be awaited for completion of populating
 * the list model.
 */
void
foundry_list_model_set_future (GListModel *model,
                               DexFuture  *future)
{
  g_return_if_fail (FOUNDRY_IS_MAIN_THREAD ());
  g_return_if_fail (G_IS_LIST_MODEL (model));
  g_return_if_fail (!future || DEX_IS_FUTURE (future));

  g_object_set_data_full (G_OBJECT (model),
                          "FOUNDRY_LIST_MODEL_FUTURE",
                          dex_ref (future),
                          dex_unref);
}

/**
 * foundry_list_model_await:
 *
 * Returns a future that resolves when the list has completed
 * being populated or rejects with error.
 *
 * Use foundry_list_model_set_future() to affect the future that is
 * used here.
 *
 * If no future has been set, this function returns a future that
 * has already resolved (e.g. True).
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_list_model_await (GListModel *model)
{
  DexFuture *future;

  dex_return_error_if_fail (FOUNDRY_IS_MAIN_THREAD ());
  dex_return_error_if_fail (G_IS_LIST_MODEL (model));

  if ((future = g_object_get_data (G_OBJECT (model), "FOUNDRY_LIST_MODEL_FUTURE")))
    return dex_ref (future);

  return dex_future_new_true ();
}
