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

#define EGG_TYPE_FLATTEN_LIST_MODEL (egg_flatten_list_model_get_type ())

G_DECLARE_FINAL_TYPE (EggFlattenListModel, egg_flatten_list_model, EGG, FLATTEN_LIST_MODEL, GObject)

EggFlattenListModel *egg_flatten_list_model_new                (GListModel          *model);
void                 egg_flatten_list_model_set_model          (EggFlattenListModel *self,
                                                                GListModel          *model);
GListModel          *egg_flatten_list_model_get_model          (EggFlattenListModel *self);
GListModel          *egg_flatten_list_model_get_model_for_item (EggFlattenListModel *self,
                                                                guint                position);

G_END_DECLS
