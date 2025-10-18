/*
 * manuals-wrapped-model.c
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

#include "manuals-wrapped-model.h"

struct _ManualsWrappedModel
{
  GObject     parent_instance;
  GListModel *model;
  DexFuture  *future;
};

static GType
manuals_wrapped_model_get_item_type (GListModel *model)
{
  return G_TYPE_OBJECT;
}

static guint
manuals_wrapped_model_get_n_items (GListModel *model)
{
  ManualsWrappedModel *self = MANUALS_WRAPPED_MODEL (model);

  if (self->model)
    return g_list_model_get_n_items (self->model);

  return 0;
}

static gpointer
manuals_wrapped_model_get_item (GListModel *model,
                                guint       position)
{
  ManualsWrappedModel *self = MANUALS_WRAPPED_MODEL (model);

  if (self->model)
    return g_list_model_get_item (self->model, position);

  return NULL;
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = manuals_wrapped_model_get_item_type;
  iface->get_n_items = manuals_wrapped_model_get_n_items;
  iface->get_item = manuals_wrapped_model_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (ManualsWrappedModel, manuals_wrapped_model, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static void
manuals_wrapped_model_finalize (GObject *object)
{
  ManualsWrappedModel *self = (ManualsWrappedModel *)object;

  dex_clear (&self->future);
  g_clear_object (&self->model);

  G_OBJECT_CLASS (manuals_wrapped_model_parent_class)->finalize (object);
}

static void
manuals_wrapped_model_class_init (ManualsWrappedModelClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = manuals_wrapped_model_finalize;
}

static void
manuals_wrapped_model_init (ManualsWrappedModel *self)
{
}

static void
manuals_wrapped_model_set_model (ManualsWrappedModel *self,
                                 GListModel          *model)
{
  g_assert (MANUALS_IS_WRAPPED_MODEL (self));
  g_assert (!model || G_IS_LIST_MODEL (model));

  if (g_set_object (&self->model, model))
    g_list_model_items_changed (G_LIST_MODEL (self),
                                0,
                                0,
                                g_list_model_get_n_items (model));
}

static DexFuture *
apply_model (DexFuture *future,
             gpointer   user_data)
{
  ManualsWrappedModel *self = user_data;
  const GValue *value;

  if ((value = dex_future_get_value (future, NULL)) &&
      G_VALUE_HOLDS (value, G_TYPE_LIST_MODEL))
    manuals_wrapped_model_set_model (self, g_value_get_object (value));

  return dex_future_new_true ();
}

GListModel *
manuals_wrapped_model_new (DexFuture *future)
{
  ManualsWrappedModel *self = g_object_new (MANUALS_TYPE_WRAPPED_MODEL, NULL);

  self->future = dex_ref (future);

  dex_future_disown (dex_future_then (dex_ref (future),
                                      apply_model,
                                      g_object_ref (self),
                                      g_object_unref));

  return G_LIST_MODEL (self);
}

DexFuture *
manuals_wrapped_model_await (ManualsWrappedModel *self)
{
  dex_return_error_if_fail (MANUALS_IS_WRAPPED_MODEL (self));

  return dex_ref (self->future);
}
