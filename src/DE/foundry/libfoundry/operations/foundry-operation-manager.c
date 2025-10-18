/* foundry-operation-manager.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include <libpeas.h>

#include "foundry-operation.h"
#include "foundry-operation-manager.h"
#include "foundry-contextual-private.h"
#include "foundry-debug.h"
#include "foundry-service-private.h"
#include "foundry-util-private.h"

struct _FoundryOperationManager
{
  FoundryService  parent_instance;
  GListStore     *operations;
};

struct _FoundryOperationManagerClass
{
  FoundryServiceClass parent_class;
};

static GType
foundry_operation_manager_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_OPERATION;
}

static guint
foundry_operation_manager_get_n_items (GListModel *model)
{
  FoundryOperationManager *self = FOUNDRY_OPERATION_MANAGER (model);

  return g_list_model_get_n_items (G_LIST_MODEL (self->operations));
}

static gpointer
foundry_operation_manager_get_item (GListModel *model,
                                    guint       position)
{
  FoundryOperationManager *self = FOUNDRY_OPERATION_MANAGER (model);

  return g_list_model_get_item (G_LIST_MODEL (self->operations), position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_n_items = foundry_operation_manager_get_n_items;
  iface->get_item_type = foundry_operation_manager_get_item_type;
  iface->get_item = foundry_operation_manager_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryOperationManager, foundry_operation_manager, FOUNDRY_TYPE_SERVICE,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static DexFuture *
foundry_operation_manager_stop (FoundryService *service)
{
  FoundryOperationManager *self = (FoundryOperationManager *)service;
  GListModel *model;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  model = G_LIST_MODEL (self->operations);
  n_items = g_list_model_get_n_items (model);

  for (guint i = n_items; i > 0; i--)
    {
      g_autoptr(FoundryOperation) operation = g_list_model_get_item (model, i - 1);

      foundry_operation_cancel (operation);
    }

  return dex_future_new_true ();
}

static void
foundry_operation_manager_finalize (GObject *object)
{
  FoundryOperationManager *self = (FoundryOperationManager *)object;

  g_clear_object (&self->operations);

  G_OBJECT_CLASS (foundry_operation_manager_parent_class)->finalize (object);
}

static void
foundry_operation_manager_class_init (FoundryOperationManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->finalize = foundry_operation_manager_finalize;

  service_class->stop = foundry_operation_manager_stop;
}

static void
foundry_operation_manager_init (FoundryOperationManager *self)
{
  self->operations = g_list_store_new (FOUNDRY_TYPE_OPERATION);

  g_signal_connect_object (self->operations,
                           "items-changed",
                           G_CALLBACK (g_list_model_items_changed),
                           self,
                           G_CONNECT_SWAPPED);
}

static void
foundry_operation_manager_remove (FoundryOperationManager *self,
                                  FoundryOperation        *operation)
{
  GListModel *model;
  guint n_items;

  g_assert (FOUNDRY_IS_OPERATION_MANAGER (self));
  g_assert (FOUNDRY_IS_OPERATION (operation));

  model = G_LIST_MODEL (self->operations);
  n_items = g_list_model_get_n_items (model);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryOperation) item = g_list_model_get_item (model, i);

      if (item == operation)
        {
          g_list_store_remove (self->operations, i);
          break;
        }
    }
}

static DexFuture *
foundry_operation_manager_done (DexFuture *completed,
                                gpointer   user_data)
{
  FoundryWeakPair *pair = user_data;
  g_autoptr(FoundryOperationManager) self = NULL;
  g_autoptr(FoundryOperation) operation = NULL;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (pair != NULL);

  foundry_weak_pair_get (pair, &self, &operation);

  if (self != NULL && operation != NULL)
    foundry_operation_manager_remove (self, operation);

  return dex_future_new_true ();
}

/**
 * foundry_operation_manager_begin:
 * @self: a #FoundryOperationManager
 * @title: a title for the operation
 *
 * Creates a new operation.
 *
 * Returns: (transfer full): a new [class@Foundry.Operation]
 */
FoundryOperation *
foundry_operation_manager_begin (FoundryOperationManager *self,
                                 const char              *title)
{
  g_autoptr(FoundryOperation) operation = NULL;
  DexFuture *future;

  g_return_val_if_fail (FOUNDRY_IS_OPERATION_MANAGER (self), NULL);

  operation = g_object_new (FOUNDRY_TYPE_OPERATION,
                            "title", title,
                            NULL);

  g_list_store_append (self->operations, operation);

  future = foundry_operation_await (operation);
  future = dex_future_finally (future,
                               foundry_operation_manager_done,
                               foundry_weak_pair_new (self, operation),
                               (GDestroyNotify)foundry_weak_pair_free);

  dex_future_disown (future);

  return g_steal_pointer (&operation);
}
