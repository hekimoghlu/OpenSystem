/* foundry-config-manager.c
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

#include "foundry-config.h"
#include "foundry-config-manager.h"
#include "foundry-config-provider-private.h"
#include "foundry-contextual-private.h"
#include "foundry-device.h"
#include "foundry-device-manager.h"
#include "foundry-debug.h"
#include "foundry-inhibitor.h"
#include "foundry-model-manager.h"
#include "foundry-sdk.h"
#include "foundry-sdk-manager.h"
#include "foundry-service-private.h"
#include "foundry-settings.h"
#include "foundry-util-private.h"

struct _FoundryConfigManager
{
  FoundryService    parent_instance;
  GListModel       *flatten;
  PeasExtensionSet *addins;
  FoundryConfig    *config;
  gsize             sequence;
};

struct _FoundryConfigManagerClass
{
  FoundryServiceClass parent_class;
};

enum {
  PROP_0,
  PROP_CONFIG,
  N_PROPS
};

static void list_model_iface_init (GListModelInterface *iface);

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryConfigManager, foundry_config_manager, FOUNDRY_TYPE_SERVICE,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static GParamSpec *properties[N_PROPS];

static void
foundry_config_manager_provider_added (PeasExtensionSet *set,
                                       PeasPluginInfo   *plugin_info,
                                       GObject          *addin,
                                       gpointer          user_data)
{
  FoundryConfigManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_CONFIG_PROVIDER (addin));
  g_assert (FOUNDRY_IS_CONFIG_MANAGER (self));

  g_debug ("Adding FoundryConfigProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_config_provider_load (FOUNDRY_CONFIG_PROVIDER (addin)));
}

static void
foundry_config_manager_provider_removed (PeasExtensionSet *set,
                                         PeasPluginInfo   *plugin_info,
                                         GObject          *addin,
                                         gpointer          user_data)
{
  FoundryConfigManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_CONFIG_PROVIDER (addin));
  g_assert (FOUNDRY_IS_CONFIG_MANAGER (self));

  g_debug ("Removing FoundryConfigProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (foundry_config_provider_unload (FOUNDRY_CONFIG_PROVIDER (addin)));
}

static void
foundry_config_manager_pick_default (FoundryConfigManager *self)
{
  g_autoptr(FoundryConfig) best_config = NULL;
  int best_priority = 0;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_CONFIG_MANAGER (self));

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryConfig) config = g_list_model_get_item (G_LIST_MODEL (self), i);
      guint priority = 0;

      if (foundry_config_can_default (config, &priority) &&
          (best_config == NULL || priority > best_priority))
        {
          g_set_object (&best_config, config);
          best_priority = priority;
        }
    }

  if (best_config)
    foundry_config_manager_set_config (self, best_config);
}

static DexFuture *
foundry_config_manager_start_fiber (gpointer user_data)
{
  FoundryConfigManager *self = user_data;
  g_autoptr(FoundrySdkManager) sdk_manager = NULL;
  g_autoptr(FoundrySettings) settings  = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryConfig) config = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autofree char *config_id = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_CONFIG_MANAGER (self));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  settings = foundry_context_load_project_settings (context);

  /* Skip if we're in a shared context */
  if (foundry_context_is_shared (context))
    return dex_future_new_true ();

  g_signal_connect_object (self->addins,
                           "extension-added",
                           G_CALLBACK (foundry_config_manager_provider_added),
                           self,
                           0);
  g_signal_connect_object (self->addins,
                           "extension-removed",
                           G_CALLBACK (foundry_config_manager_provider_removed),
                           self,
                           0);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryConfigProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures,
                       foundry_config_provider_load (provider));
    }

  if (futures->len > 0)
    dex_await (foundry_future_all (futures), NULL);

  /* Wait for SDK manager to be started */
  sdk_manager = foundry_context_dup_sdk_manager (context);
  dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (sdk_manager)), NULL);

  /* Apply config from last session */
  config_id = foundry_settings_get_string (settings, "config");
  if ((config = foundry_config_manager_find_config (self, config_id)))
    foundry_config_manager_set_config (self, config);

  /* If we have no config pick a default config */
  if (self->config == NULL)
    foundry_config_manager_pick_default (self);

  return dex_future_new_true ();
}

static DexFuture *
foundry_config_manager_start (FoundryService *service)
{
  FoundryConfigManager *self = (FoundryConfigManager *)service;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_CONFIG_MANAGER (self));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_config_manager_start_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static DexFuture *
foundry_config_manager_stop (FoundryService *service)
{
  FoundryConfigManager *self = (FoundryConfigManager *)service;
  g_autoptr(FoundrySettings) settings = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  settings = foundry_context_load_project_settings (context);

  if (self->config != NULL)
    {
      g_autofree char *id = foundry_config_dup_id (self->config);

      foundry_settings_set_string (settings, "config", id);
    }

  g_clear_object (&self->config);

  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_config_manager_provider_added),
                                        self);
  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_config_manager_provider_removed),
                                        self);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryConfigProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures,
                       foundry_config_provider_unload (provider));
    }

  g_clear_object (&self->addins);

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static void
foundry_config_manager_constructed (GObject *object)
{
  FoundryConfigManager *self = (FoundryConfigManager *)object;
  g_autoptr(FoundryContext) context = NULL;

  G_OBJECT_CLASS (foundry_config_manager_parent_class)->constructed (object);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->addins = peas_extension_set_new (NULL,
                                         FOUNDRY_TYPE_CONFIG_PROVIDER,
                                         "context", context,
                                         NULL);

  g_object_set (self->flatten,
                "model", self->addins,
                NULL);
}

static void
foundry_config_manager_finalize (GObject *object)
{
  FoundryConfigManager *self = (FoundryConfigManager *)object;

  g_clear_object (&self->flatten);
  g_clear_object (&self->config);
  g_clear_object (&self->addins);

  G_OBJECT_CLASS (foundry_config_manager_parent_class)->finalize (object);
}

static void
foundry_config_manager_get_property (GObject    *object,
                                     guint       prop_id,
                                     GValue     *value,
                                     GParamSpec *pspec)
{
  FoundryConfigManager *self = FOUNDRY_CONFIG_MANAGER (object);

  switch (prop_id)
    {
    case PROP_CONFIG:
      g_value_take_object (value, foundry_config_manager_dup_config (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_config_manager_set_property (GObject      *object,
                                     guint         prop_id,
                                     const GValue *value,
                                     GParamSpec   *pspec)
{
  FoundryConfigManager *self = FOUNDRY_CONFIG_MANAGER (object);

  switch (prop_id)
    {
    case PROP_CONFIG:
      foundry_config_manager_set_config (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_config_manager_class_init (FoundryConfigManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->constructed = foundry_config_manager_constructed;
  object_class->finalize = foundry_config_manager_finalize;
  object_class->get_property = foundry_config_manager_get_property;
  object_class->set_property = foundry_config_manager_set_property;

  service_class->start = foundry_config_manager_start;
  service_class->stop = foundry_config_manager_stop;

  properties[PROP_CONFIG] =
    g_param_spec_object ("config", NULL, NULL,
                         FOUNDRY_TYPE_CONFIG,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_config_manager_init (FoundryConfigManager *self)
{
  self->flatten = foundry_flatten_list_model_new (NULL);

  g_signal_connect_object (self->flatten,
                           "items-changed",
                           G_CALLBACK (g_list_model_items_changed),
                           self,
                           G_CONNECT_SWAPPED);
}

static GType
foundry_config_manager_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_CONFIG;
}

static guint
foundry_config_manager_get_n_items (GListModel *model)
{
  return g_list_model_get_n_items (G_LIST_MODEL (FOUNDRY_CONFIG_MANAGER (model)->flatten));
}

static gpointer
foundry_config_manager_get_item (GListModel *model,
                                 guint       position)
{
  return g_list_model_get_item (G_LIST_MODEL (FOUNDRY_CONFIG_MANAGER (model)->flatten), position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_config_manager_get_item_type;
  iface->get_n_items = foundry_config_manager_get_n_items;
  iface->get_item = foundry_config_manager_get_item;
}

/**
 * foundry_config_manager_dup_config:
 * @self: a #FoundryConfigManager
 *
 * Gets the active configuration
 *
 * Returns: (transfer full) (nullable): a #FoundryConfig or %NULL
 */
FoundryConfig *
foundry_config_manager_dup_config (FoundryConfigManager *self)
{
  FoundryConfig *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_CONFIG_MANAGER (self), NULL);

  g_set_object (&ret, self->config);

  return ret;
}

typedef struct _Apply
{
  FoundryConfigManager *self;
  FoundrySdkManager    *sdk_manager;
  FoundryConfig        *config;
  FoundryDevice        *device;
  FoundryInhibitor     *inhibitor;
  gsize                 sequence;
} Apply;

static void
apply_free (Apply *state)
{
  g_clear_object (&state->self);
  g_clear_object (&state->config);
  g_clear_object (&state->device);
  g_clear_object (&state->sdk_manager);
  g_clear_object (&state->inhibitor);
  g_free (state);
}

static DexFuture *
foundry_config_manager_apply_fiber (gpointer data)
{
  Apply *state = data;
  g_autoptr(FoundrySdk) sdk = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_CONFIG_MANAGER (state->self));
  g_assert (FOUNDRY_IS_CONFIG (state->config));

  if (state->sdk_manager == NULL || state->device == NULL)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_INVAL,
                                  "Cannot apply configuration");

  if ((sdk = dex_await_object (foundry_config_resolve_sdk (state->config,
                                                           state->device),
                               &error)))
    {
      if (state->sequence == state->self->sequence)
        foundry_sdk_manager_set_sdk (state->sdk_manager, sdk);

      return dex_future_new_true ();
    }

  return dex_future_new_for_error (g_steal_pointer (&error));
}

static void
foundry_config_manager_apply (FoundryConfigManager *self,
                              FoundryConfig        *config)
{
  Apply *state;
  g_autoptr(FoundryDeviceManager) device_manager = NULL;
  g_autoptr(FoundrySdkManager) sdk_manager = NULL;
  g_autoptr(FoundryInhibitor) inhibitor = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundrySdk) sdk = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_CONFIG_MANAGER (self));
  g_assert (FOUNDRY_IS_CONFIG (config));

  if (!(inhibitor = foundry_contextual_inhibit (FOUNDRY_CONTEXTUAL (self), NULL)))
    return;

  context = foundry_inhibitor_dup_context (inhibitor);
  device_manager = foundry_context_dup_device_manager (context);

  state = g_new0 (Apply, 1);
  state->sequence = ++self->sequence;
  state->self = g_object_ref (self);
  state->config = g_object_ref (config);
  state->device = foundry_device_manager_dup_device (device_manager);
  state->sdk_manager = foundry_context_dup_sdk_manager (context);
  state->inhibitor = g_steal_pointer (&inhibitor);

  dex_future_disown (dex_scheduler_spawn (NULL, 0,
                                          foundry_config_manager_apply_fiber,
                                          state,
                                          (GDestroyNotify) apply_free));
}

/**
 * foundry_config_manager_set_config:
 * @self: a #FoundryConfigManager
 *
 * Sets the active configuration for the config manager.
 *
 * Other services such as the build manager or sdk manager may respond
 * to changes of this property and update accordingly.
 */
void
foundry_config_manager_set_config (FoundryConfigManager *self,
                                   FoundryConfig        *config)
{
  g_autoptr(FoundrySettings) settings = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryConfig) old = NULL;
  g_autofree char *config_id = NULL;

  g_return_if_fail (FOUNDRY_IS_CONFIG_MANAGER (self));
  g_return_if_fail (!config || FOUNDRY_IS_CONFIG (config));

  if (self->config == config)
    return;

  if (config != NULL)
    {
      config_id = foundry_config_dup_id (config);
      g_object_ref (config);
    }

  old = g_steal_pointer (&self->config);
  self->config = config;

  if (old != NULL)
    g_object_notify (G_OBJECT (old), "active");

  if (config != NULL)
    foundry_config_manager_apply (self, config);

  _foundry_contextual_invalidate_pipeline (FOUNDRY_CONTEXTUAL (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  settings = foundry_context_load_project_settings (context);
  foundry_settings_set_string (settings, "config", config_id ? config_id : "");

  g_object_notify (G_OBJECT (context), "build-system");

  if (config != NULL)
    g_object_notify (G_OBJECT (config), "active");

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CONFIG]);
}

/**
 * foundry_config_manager_find_config:
 * @self: a #FoundryConfigManager
 * @config_id: an identifier matching a #FoundryConfig:id
 *
 * Looks through available configs to find one matching @config_id.
 *
 * Returns: (transfer full) (nullable): a #FoundryConfig or %NULL
 */
FoundryConfig *
foundry_config_manager_find_config (FoundryConfigManager *self,
                                    const char           *config_id)
{
  guint n_items;

  g_return_val_if_fail (FOUNDRY_IS_CONFIG_MANAGER (self), NULL);
  g_return_val_if_fail (config_id != NULL, NULL);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryConfig) config = g_list_model_get_item (G_LIST_MODEL (self), i);
      g_autofree char *id = foundry_config_dup_id (config);

      if (g_strcmp0 (config_id, id) == 0)
        return g_steal_pointer (&config);
    }

  return NULL;
}
