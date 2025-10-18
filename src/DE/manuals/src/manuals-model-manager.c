/*
 * manuals-model-manager.c
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "config.h"

#include <gtk/gtk.h>

#include "manuals-model-manager.h"

struct _ManualsModelManager
{
  FoundryModelManager parent_instance;
};

G_DEFINE_FINAL_TYPE (ManualsModelManager, manuals_model_manager, FOUNDRY_TYPE_MODEL_MANAGER)

static GListModel *
manuals_model_manager_flatten (FoundryModelManager *self,
                               GListModel          *model)
{
  return G_LIST_MODEL (gtk_flatten_list_model_new (model));
}

static GListModel *
manuals_model_manager_map (FoundryModelManager     *self,
                           GListModel              *model,
                           FoundryListModelMapFunc  map_func,
                           gpointer                 user_data,
                           GDestroyNotify           user_destroy)
{
  return G_LIST_MODEL (gtk_map_list_model_new (model, (gpointer)map_func, user_data, user_destroy));
}

static void
manuals_model_manager_class_init (ManualsModelManagerClass *klass)
{
  FoundryModelManagerClass *model_manager_class = FOUNDRY_MODEL_MANAGER_CLASS (klass);

  model_manager_class->flatten = manuals_model_manager_flatten;
  model_manager_class->map = manuals_model_manager_map;
}

static void
manuals_model_manager_init (ManualsModelManager *self)
{
}

FoundryModelManager *
manuals_model_manager_new (void)
{
  return g_object_new (MANUALS_TYPE_MODEL_MANAGER, NULL);
}
