/* foundry-model-manager.h
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

#pragma once

#include <libdex.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_MODEL_MANAGER (foundry_model_manager_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryModelManager, foundry_model_manager, FOUNDRY, MODEL_MANAGER, GObject)

/**
 * FoundryListModelMapFunc:
 * @item: (transfer full) (type GObject): The item to map
 * @user_data: User data
 *
 * Returns: (transfer full):
 */
typedef gpointer (*FoundryListModelMapFunc) (gpointer item,
                                             gpointer user_data);

struct _FoundryModelManagerClass
{
  GObjectClass parent_class;

  GListModel *(*flatten) (FoundryModelManager     *self,
                          GListModel              *model);
  GListModel *(*map)     (FoundryModelManager     *self,
                          GListModel              *model,
                          FoundryListModelMapFunc  map_func,
                          gpointer                 user_data,
                          GDestroyNotify           user_destroy);

  /*< private >*/
  gpointer _reserved[13];
};

FOUNDRY_AVAILABLE_IN_ALL
FoundryModelManager *foundry_model_manager_dup_default (void);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_model_manager_set_default (FoundryModelManager     *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel          *foundry_model_manager_flatten     (FoundryModelManager     *self,
                                                        GListModel              *model);
FOUNDRY_AVAILABLE_IN_ALL
GListModel          *foundry_model_manager_map         (FoundryModelManager     *self,
                                                        GListModel              *model,
                                                        FoundryListModelMapFunc  map_func,
                                                        gpointer                 user_data,
                                                        GDestroyNotify           user_destroy);
FOUNDRY_AVAILABLE_IN_ALL
GListModel          *foundry_flatten_list_model_new    (GListModel              *model);
FOUNDRY_AVAILABLE_IN_ALL
GListModel          *foundry_map_list_model_new        (GListModel              *model,
                                                        FoundryListModelMapFunc  map_func,
                                                        gpointer                 user_data,
                                                        GDestroyNotify           user_destroy);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_list_model_set_future     (GListModel              *model,
                                                        DexFuture               *future);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_list_model_await          (GListModel              *model);

G_END_DECLS
