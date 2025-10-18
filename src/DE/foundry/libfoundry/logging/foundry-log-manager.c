/* foundry-log-manager.c
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

#include "foundry-contextual-private.h"
#include "foundry-debug.h"
#include "foundry-log-model-private.h"
#include "foundry-log-manager-private.h"
#include "foundry-log-message.h"
#include "foundry-service-private.h"
#include "foundry-util-private.h"

struct _FoundryLogManager
{
  FoundryService   parent_instance;
  FoundryLogModel *log_model;
};

struct _FoundryLogManagerClass
{
  FoundryServiceClass parent_class;
};

static void list_model_iface_init (GListModelInterface *iface);

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryLogManager, foundry_log_manager, FOUNDRY_TYPE_SERVICE,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static DexFuture *
foundry_log_manager_start (FoundryService *service)
{
  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  return dex_future_new_true ();
}

static DexFuture *
foundry_log_manager_stop (FoundryService *service)
{
  FoundryLogManager *self = (FoundryLogManager *)service;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  _foundry_log_model_remove_all (self->log_model);

  return dex_future_new_true ();
}

static void
foundry_log_manager_finalize (GObject *object)
{
  FoundryLogManager *self = (FoundryLogManager *)object;

  g_clear_object (&self->log_model);

  G_OBJECT_CLASS (foundry_log_manager_parent_class)->finalize (object);
}

static void
foundry_log_manager_class_init (FoundryLogManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->finalize = foundry_log_manager_finalize;

  service_class->start = foundry_log_manager_start;
  service_class->stop = foundry_log_manager_stop;
}

static void
foundry_log_manager_init (FoundryLogManager *self)
{
  self->log_model = _foundry_log_model_new ();
}

static GType
foundry_log_manager_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_LOG_MESSAGE;
}

static guint
foundry_log_manager_get_n_items (GListModel *model)
{
  FoundryLogManager *self = FOUNDRY_LOG_MANAGER (model);

  return g_list_model_get_n_items (G_LIST_MODEL (self->log_model));
}

static gpointer
foundry_log_manager_get_item (GListModel *model,
                              guint       position)
{
  FoundryLogManager *self = FOUNDRY_LOG_MANAGER (model);

  return g_list_model_get_item (G_LIST_MODEL (self->log_model), position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_log_manager_get_item_type;
  iface->get_n_items = foundry_log_manager_get_n_items;
  iface->get_item = foundry_log_manager_get_item;
}

void
_foundry_log_manager_append (FoundryLogManager *self,
                             const char        *domain,
                             GLogLevelFlags     severity,
                             char              *message)
{
  _foundry_log_model_append (self->log_model, domain, severity, message);
}
