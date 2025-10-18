/* foundry-sdk-manager.c
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

#include "foundry-contextual-private.h"
#include "foundry-debug.h"
#include "foundry-model-manager.h"
#include "foundry-sdk-manager.h"
#include "foundry-sdk-provider-private.h"
#include "foundry-sdk.h"
#include "foundry-service-private.h"
#include "foundry-util-private.h"

struct _FoundrySdkManager
{
  FoundryService    parent_instance;
  GListModel       *flatten;
  PeasExtensionSet *addins;
  FoundrySdk       *sdk;
};

struct _FoundrySdkManagerClass
{
  FoundryServiceClass parent_class;
};

enum {
  PROP_0,
  PROP_SDK,
  N_PROPS
};

static void list_model_iface_init (GListModelInterface *iface);

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundrySdkManager, foundry_sdk_manager, FOUNDRY_TYPE_SERVICE,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static GParamSpec *properties[N_PROPS];

static void
foundry_sdk_manager_provider_added (PeasExtensionSet *set,
                                    PeasPluginInfo   *plugin_info,
                                    GObject          *addin,
                                    gpointer          user_data)
{
  FoundrySdkManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_SDK_PROVIDER (addin));
  g_assert (FOUNDRY_IS_SDK_MANAGER (self));

  g_debug ("Adding FoundrySdkProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_sdk_provider_load (FOUNDRY_SDK_PROVIDER (addin)));
}

static void
foundry_sdk_manager_provider_removed (PeasExtensionSet *set,
                                      PeasPluginInfo   *plugin_info,
                                      GObject          *addin,
                                      gpointer          user_data)
{
  FoundrySdkManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_SDK_PROVIDER (addin));
  g_assert (FOUNDRY_IS_SDK_MANAGER (self));

  g_debug ("Removing FoundrySdkProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_sdk_provider_unload (FOUNDRY_SDK_PROVIDER (addin)));
}

static DexFuture *
foundry_sdk_manager_start (FoundryService *service)
{
  FoundrySdkManager *self = (FoundrySdkManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  g_signal_connect_object (self->addins,
                           "extension-added",
                           G_CALLBACK (foundry_sdk_manager_provider_added),
                           self,
                           0);
  g_signal_connect_object (self->addins,
                           "extension-removed",
                           G_CALLBACK (foundry_sdk_manager_provider_removed),
                           self,
                           0);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundrySdkProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures,
                       foundry_sdk_provider_load (provider));
    }

  if (futures->len > 0)
    return dex_future_catch (foundry_future_all (futures),
                             foundry_log_rejections,
                             NULL, NULL);

  return dex_future_new_true ();
}

static DexFuture *
foundry_sdk_manager_stop (FoundryService *service)
{
  FoundrySdkManager *self = (FoundrySdkManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  g_clear_object (&self->sdk);

  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_sdk_manager_provider_added),
                                        self);
  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_sdk_manager_provider_removed),
                                        self);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundrySdkProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures,
                       foundry_sdk_provider_unload (provider));
    }

  g_clear_object (&self->addins);

  if (futures->len > 0)
    return dex_future_catch (foundry_future_all (futures),
                             foundry_log_rejections,
                             NULL, NULL);

  return dex_future_new_true ();
}

static void
foundry_sdk_manager_constructed (GObject *object)
{
  FoundrySdkManager *self = (FoundrySdkManager *)object;
  g_autoptr(FoundryContext) context = NULL;

  G_OBJECT_CLASS (foundry_sdk_manager_parent_class)->constructed (object);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->addins = peas_extension_set_new (NULL,
                                         FOUNDRY_TYPE_SDK_PROVIDER,
                                         "context", context,
                                         NULL);

  g_object_set (self->flatten,
                "model", self->addins,
                NULL);
}

static void
foundry_sdk_manager_finalize (GObject *object)
{
  FoundrySdkManager *self = (FoundrySdkManager *)object;

  g_clear_object (&self->flatten);
  g_clear_object (&self->sdk);
  g_clear_object (&self->addins);

  G_OBJECT_CLASS (foundry_sdk_manager_parent_class)->finalize (object);
}

static void
foundry_sdk_manager_get_property (GObject    *object,
                                  guint       prop_id,
                                  GValue     *value,
                                  GParamSpec *pspec)
{
  FoundrySdkManager *self = FOUNDRY_SDK_MANAGER (object);

  switch (prop_id)
    {
    case PROP_SDK:
      g_value_take_object (value, foundry_sdk_manager_dup_sdk (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_sdk_manager_set_property (GObject      *object,
                                  guint         prop_id,
                                  const GValue *value,
                                  GParamSpec   *pspec)
{
  FoundrySdkManager *self = FOUNDRY_SDK_MANAGER (object);

  switch (prop_id)
    {
    case PROP_SDK:
      foundry_sdk_manager_set_sdk (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_sdk_manager_class_init (FoundrySdkManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->constructed = foundry_sdk_manager_constructed;
  object_class->finalize = foundry_sdk_manager_finalize;
  object_class->get_property = foundry_sdk_manager_get_property;
  object_class->set_property = foundry_sdk_manager_set_property;

  service_class->start = foundry_sdk_manager_start;
  service_class->stop = foundry_sdk_manager_stop;

  properties[PROP_SDK] =
    g_param_spec_object ("sdk", NULL, NULL,
                         FOUNDRY_TYPE_SDK,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_sdk_manager_init (FoundrySdkManager *self)
{
  self->flatten = foundry_flatten_list_model_new (NULL);

  g_signal_connect_object (self->flatten,
                           "items-changed",
                           G_CALLBACK (g_list_model_items_changed),
                           self,
                           G_CONNECT_SWAPPED);
}

static GType
foundry_sdk_manager_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_SDK;
}

static guint
foundry_sdk_manager_get_n_items (GListModel *model)
{
  return g_list_model_get_n_items (G_LIST_MODEL (FOUNDRY_SDK_MANAGER (model)->flatten));
}

static gpointer
foundry_sdk_manager_get_item (GListModel *model,
                              guint       position)
{
  return g_list_model_get_item (G_LIST_MODEL (FOUNDRY_SDK_MANAGER (model)->flatten), position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_sdk_manager_get_item_type;
  iface->get_n_items = foundry_sdk_manager_get_n_items;
  iface->get_item = foundry_sdk_manager_get_item;
}

/**
 * foundry_sdk_manager_dup_sdk:
 * @self: a #FoundrySdkManager
 *
 * Gets the active SDK.
 *
 * This is generally used for build pipelines, terminal shells, and more.
 *
 * Returns: (transfer full) (nullable): a #FoundrySdk or %NULL
 */
FoundrySdk *
foundry_sdk_manager_dup_sdk (FoundrySdkManager *self)
{
  FoundrySdk *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_SDK_MANAGER (self), NULL);

  g_set_object (&ret, self->sdk);

  return ret;
}

void
foundry_sdk_manager_set_sdk (FoundrySdkManager *self,
                             FoundrySdk        *sdk)
{
  g_autoptr(FoundrySdk) old = NULL;

  g_return_if_fail (FOUNDRY_IS_SDK_MANAGER (self));
  g_return_if_fail (!sdk || FOUNDRY_IS_SDK (sdk));

  if (self->sdk == sdk)
    return;

  if (sdk)
    g_object_ref (sdk);

  old = g_steal_pointer (&self->sdk);
  self->sdk = sdk;

  if (old != NULL)
    g_object_notify (G_OBJECT (old), "active");

  if (sdk != NULL)
    g_object_notify (G_OBJECT (sdk), "active");

  _foundry_contextual_invalidate_pipeline (FOUNDRY_CONTEXTUAL (self));
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SDK]);
}

/**
 * foundry_sdk_manager_find_by_id:
 * @self: a [class@Foundry.SdkManager]
 * @sdk_id: the identifier of the SDK
 *
 * Find a SDK by its identifier.
 *
 * Returns: (transfer full) (nullable): a [class@Foundry.Sdk] or %NULL
 */
DexFuture *
foundry_sdk_manager_find_by_id (FoundrySdkManager *self,
                                const char        *sdk_id)
{
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  dex_return_error_if_fail (FOUNDRY_IS_SDK_MANAGER (self));
  dex_return_error_if_fail (sdk_id != NULL);
  dex_return_error_if_fail (self->addins != NULL);

  futures = g_ptr_array_new_with_free_func (dex_unref);
  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundrySdkProvider) sdk_provider = NULL;
      DexFuture *future;

      sdk_provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);
      future = foundry_sdk_provider_find_by_id (sdk_provider, sdk_id);

      g_ptr_array_add (futures, g_steal_pointer (&future));
    }

  if (futures->len == 0)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "Not found");

  return dex_future_anyv ((DexFuture **)futures->pdata, futures->len);
}
