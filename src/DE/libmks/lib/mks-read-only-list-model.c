/*
 * mks-read-only-list-model.c
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "mks-read-only-list-model-private.h"

struct _MksReadOnlyListModel
{
  GObject parent_instance;
  GListModel *model;
  gulong items_changed_handler;
};

static GType
mks_read_only_list_model_get_item_type (GListModel *model)
{
  MksReadOnlyListModel *self = MKS_READ_ONLY_LIST_MODEL (model);

  if (self->model != NULL)
    return g_list_model_get_item_type (self->model);

  return G_TYPE_OBJECT;
}

static guint
mks_read_only_list_model_get_n_items (GListModel *model)
{
  MksReadOnlyListModel *self = MKS_READ_ONLY_LIST_MODEL (model);

  if (self->model != NULL)
    return g_list_model_get_n_items (self->model);

  return 0;
}

static gpointer
mks_read_only_list_model_get_item (GListModel *model,
                                   guint       position)
{
  MksReadOnlyListModel *self = MKS_READ_ONLY_LIST_MODEL (model);

  if (self->model != NULL)
    return g_list_model_get_item (self->model, position);

  return NULL;
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = mks_read_only_list_model_get_item_type;
  iface->get_n_items = mks_read_only_list_model_get_n_items;
  iface->get_item = mks_read_only_list_model_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (MksReadOnlyListModel, mks_read_only_list_model, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

enum {
  PROP_0,
  PROP_N_ITEMS,
  N_PROPS
};

static GParamSpec *properties [N_PROPS];

static void
mks_read_only_list_model_items_changed_cb (MksReadOnlyListModel *self,
                                           guint                 position,
                                           guint                 removed,
                                           guint                 added,
                                           GListModel           *model)
{
  g_assert (MKS_IS_READ_ONLY_LIST_MODEL (self));
  g_assert (G_IS_LIST_MODEL (model));

  if (removed == 0 && added == 0)
    return;

  g_list_model_items_changed (G_LIST_MODEL (self), position, removed, added);

  if (removed != 0 || added != 0)
    g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_N_ITEMS]);
}

/**
 * mks_read_only_list_model_new:
 * @model: a #GListModel
 *
 * Creates a #GListModel which is read-only.
 *
 * This is useful for situations where you want to allow for observation
 * of a list but restrict API access to the underlying #GListModel.
 *
 * Returns: (transfer full): a new #MksReadOnlyListModel
 */
GListModel *
mks_read_only_list_model_new (GListModel *model)
{
  MksReadOnlyListModel *self;

  g_return_val_if_fail (G_IS_LIST_MODEL (model), NULL);

  self = g_object_new (MKS_TYPE_READ_ONLY_LIST_MODEL, NULL);

  if (g_set_object (&self->model, model))
    self->items_changed_handler =
      g_signal_connect_object (self->model,
                               "items-changed",
                               G_CALLBACK (mks_read_only_list_model_items_changed_cb),
                               self,
                               G_CONNECT_SWAPPED);

  return G_LIST_MODEL (self);
}

static void
mks_read_only_list_model_dispose (GObject *object)
{
  MksReadOnlyListModel *self = (MksReadOnlyListModel *)object;

  if (self->model != NULL)
    {
      g_clear_signal_handler (&self->items_changed_handler, self->model);
      g_clear_object (&self->model);
    }

  G_OBJECT_CLASS (mks_read_only_list_model_parent_class)->dispose (object);
}

static void
mks_read_only_list_model_get_property (GObject    *object,
                                       guint       prop_id,
                                       GValue     *value,
                                       GParamSpec *pspec)
{
  MksReadOnlyListModel *self = MKS_READ_ONLY_LIST_MODEL (object);

  switch (prop_id)
    {
    case PROP_N_ITEMS:
      g_value_set_uint (value, self->model ? g_list_model_get_n_items (self->model) : 0);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
mks_read_only_list_model_class_init (MksReadOnlyListModelClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = mks_read_only_list_model_dispose;
  object_class->get_property = mks_read_only_list_model_get_property;

  /**
   * MksReadOnlyListModel:n-items:
   *
   * The number of items in the underlying model.
   *
   * This is useful for binding in GtkBuilder UI definitions so that widgetry
   * may be automatically hidden when the list is empty.
   */
  properties [PROP_N_ITEMS] =
    g_param_spec_uint ("n-items", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
mks_read_only_list_model_init (MksReadOnlyListModel *self)
{
}
