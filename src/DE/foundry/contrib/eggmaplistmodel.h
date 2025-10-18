/*
 * Copyright © 2018 Benjamin Otte
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Benjamin Otte <otte@gnome.org>
 */

#pragma once

#include <gio/gio.h>

G_BEGIN_DECLS

#define EGG_TYPE_MAP_LIST_MODEL (egg_map_list_model_get_type ())

G_DECLARE_FINAL_TYPE (EggMapListModel, egg_map_list_model, EGG, MAP_LIST_MODEL, GObject)

/**
 * EggMapListModelMapFunc:
 * @item: (type GObject) (transfer full): The item to map
 * @user_data: user data
 *
 * User function that is called to map an @item of the original model to
 * an item expected by the map model.
 *
 * The returned items must conform to the item type of the model they are
 * used with.
 *
 * Returns: (type GObject) (transfer full): The item to map to
 */
typedef gpointer (* EggMapListModelMapFunc) (gpointer item, gpointer user_data);

EggMapListModel *       egg_map_list_model_new                  (GListModel             *model,
                                                                 EggMapListModelMapFunc  map_func,
                                                                 gpointer                user_data,
                                                                 GDestroyNotify          user_destroy);

void                    egg_map_list_model_set_map_func         (EggMapListModel        *self,
                                                                 EggMapListModelMapFunc  map_func,
                                                                 gpointer                user_data,
                                                                 GDestroyNotify          user_destroy);
void                    egg_map_list_model_set_model            (EggMapListModel        *self,
                                                                 GListModel             *model);
GListModel *            egg_map_list_model_get_model            (EggMapListModel        *self);
gboolean                egg_map_list_model_has_map              (EggMapListModel        *self);

G_END_DECLS
