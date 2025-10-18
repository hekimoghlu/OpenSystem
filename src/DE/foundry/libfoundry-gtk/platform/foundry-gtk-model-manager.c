/* foundry-gtk-model-manager.c
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

#include <gtk/gtk.h>

#include "foundry-gtk-model-manager-private.h"

struct _FoundryGtkModelManager
{
  FoundryModelManager parent_instance;
};

G_DEFINE_FINAL_TYPE (FoundryGtkModelManager, foundry_gtk_model_manager, FOUNDRY_TYPE_MODEL_MANAGER)

static GListModel *
foundry_model_manager_real_flatten (FoundryModelManager *self,
                                    GListModel          *model)
{
  return G_LIST_MODEL (gtk_flatten_list_model_new (model));
}

static GListModel *
foundry_model_manager_real_map (FoundryModelManager     *self,
                                GListModel              *model,
                                FoundryListModelMapFunc  map_func,
                                gpointer                 user_data,
                                GDestroyNotify           user_destroy)
{
  return G_LIST_MODEL (gtk_map_list_model_new (model, map_func, user_data, user_destroy));
}

static void
foundry_gtk_model_manager_class_init (FoundryGtkModelManagerClass *klass)
{
  FoundryModelManagerClass *model_manager_class = FOUNDRY_MODEL_MANAGER_CLASS (klass);

  model_manager_class->flatten = foundry_model_manager_real_flatten;
  model_manager_class->map = foundry_model_manager_real_map;
}

static void
foundry_gtk_model_manager_init (FoundryGtkModelManager *self)
{
}
