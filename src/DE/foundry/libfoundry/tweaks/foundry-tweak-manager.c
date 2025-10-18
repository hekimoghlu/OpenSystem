/* foundry-tweak-manager.c
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

#include <libpeas.h>

#include "foundry-contextual-private.h"
#include "foundry-debug.h"
#include "foundry-model-manager.h"
#include "foundry-tweak-provider-private.h"
#include "foundry-tweak-manager.h"
#include "foundry-service-private.h"
#include "foundry-util-private.h"

struct _FoundryTweakManager
{
  FoundryService    parent_instance;
  PeasExtensionSet *addins;
  FoundryTweakTree *tree;
};

struct _FoundryTweakManagerClass
{
  FoundryServiceClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryTweakManager, foundry_tweak_manager, FOUNDRY_TYPE_SERVICE)

static void
foundry_tweak_manager_provider_added (PeasExtensionSet *set,
                                      PeasPluginInfo   *plugin_info,
                                      GObject          *addin,
                                      gpointer          user_data)
{
  FoundryTweakManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_TWEAK_PROVIDER (addin));
  g_assert (FOUNDRY_IS_TWEAK_MANAGER (self));

  g_debug ("Adding FoundryTweakProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (_foundry_tweak_provider_load (FOUNDRY_TWEAK_PROVIDER (addin), self->tree));
}

static void
foundry_tweak_manager_provider_removed (PeasExtensionSet *set,
                                        PeasPluginInfo   *plugin_info,
                                        GObject          *addin,
                                        gpointer          user_data)
{
  FoundryTweakManager *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_TWEAK_PROVIDER (addin));
  g_assert (FOUNDRY_IS_TWEAK_MANAGER (self));

  g_debug ("Removing FoundryTweakProvider of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (_foundry_tweak_provider_unload (FOUNDRY_TWEAK_PROVIDER (addin)));
}

static DexFuture *
foundry_tweak_manager_start (FoundryService *service)
{
  FoundryTweakManager *self = (FoundryTweakManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));
  g_assert (PEAS_IS_EXTENSION_SET (self->addins));

  g_signal_connect_object (self->addins,
                           "extension-added",
                           G_CALLBACK (foundry_tweak_manager_provider_added),
                           self,
                           0);
  g_signal_connect_object (self->addins,
                           "extension-removed",
                           G_CALLBACK (foundry_tweak_manager_provider_removed),
                           self,
                           0);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryTweakProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, _foundry_tweak_provider_load (provider, self->tree));
    }

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static DexFuture *
foundry_tweak_manager_stop (FoundryService *service)
{
  FoundryTweakManager *self = (FoundryTweakManager *)service;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SERVICE (service));

  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_tweak_manager_provider_added),
                                        self);
  g_signal_handlers_disconnect_by_func (self->addins,
                                        G_CALLBACK (foundry_tweak_manager_provider_removed),
                                        self);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));
  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryTweakProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, _foundry_tweak_provider_unload (provider));
    }

  g_clear_object (&self->addins);

  if (futures->len > 0)
    return foundry_future_all (futures);

  return dex_future_new_true ();
}

static void
foundry_tweak_manager_constructed (GObject *object)
{
  FoundryTweakManager *self = (FoundryTweakManager *)object;
  g_autoptr(FoundryContext) context = NULL;

  G_OBJECT_CLASS (foundry_tweak_manager_parent_class)->constructed (object);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->tree = foundry_tweak_tree_new (context);
  self->addins = peas_extension_set_new (NULL,
                                         FOUNDRY_TYPE_TWEAK_PROVIDER,
                                         "context", context,
                                         NULL);
}

static void
foundry_tweak_manager_dispose (GObject *object)
{
  FoundryTweakManager *self = (FoundryTweakManager *)object;

  g_clear_object (&self->addins);
  g_clear_object (&self->tree);

  G_OBJECT_CLASS (foundry_tweak_manager_parent_class)->dispose (object);
}

static void
foundry_tweak_manager_class_init (FoundryTweakManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->constructed = foundry_tweak_manager_constructed;
  object_class->dispose = foundry_tweak_manager_dispose;

  service_class->start = foundry_tweak_manager_start;
  service_class->stop = foundry_tweak_manager_stop;
}

static void
foundry_tweak_manager_init (FoundryTweakManager *self)
{
}

static DexFuture *
foundry_tweak_manager_list_children_cb (DexFuture *future,
                                        gpointer   user_data)
{
  g_autoptr(FoundryTweakManager) self = dex_await_object (dex_ref (future), NULL);
  const char *path = user_data;

  g_assert (FOUNDRY_IS_TWEAK_MANAGER (self));
  g_assert (path != NULL);

  return foundry_tweak_tree_list (self->tree, path);
}

/**
 * foundry_tweak_manager_list_children:
 * @self: a [class@Foundry.TweakManager]
 * @path: the tweaks path
 *
 * Returns: (transfer full) (nullable): A [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] of [class@Foundry.Tweak].
 *
 */
DexFuture *
foundry_tweak_manager_list_children (FoundryTweakManager *self,
                                     const char          *path)
{
  DexFuture *future;

  dex_return_error_if_fail (FOUNDRY_IS_TWEAK_MANAGER (self));
  dex_return_error_if_fail (path != NULL);

  future = foundry_service_when_ready (FOUNDRY_SERVICE (self));
  future = dex_future_then (future,
                            foundry_future_return_object,
                            g_object_ref (self),
                            g_object_unref);
  future = dex_future_then (future,
                            foundry_tweak_manager_list_children_cb,
                            g_strdup (path),
                            g_free);

  return future;
}
