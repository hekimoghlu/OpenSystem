/* foundry-dependency-provider.c
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

#include "foundry-config.h"
#include "foundry-dependency.h"
#include "foundry-dependency-provider-private.h"

typedef struct
{
  PeasPluginInfo *plugin_info;
} FoundryDependencyProviderPrivate;

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryDependencyProvider, foundry_dependency_provider, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_PLUGIN_INFO,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static DexFuture *
foundry_dependency_provider_real_list_dependencies (FoundryDependencyProvider *self,
                                                    FoundryConfig             *config,
                                                    FoundryDependency         *dependency)
{
  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Listing dependencies is not supported");
}

static void
foundry_dependency_provider_finalize (GObject *object)
{
  FoundryDependencyProvider *self = (FoundryDependencyProvider *)object;
  FoundryDependencyProviderPrivate *priv = foundry_dependency_provider_get_instance_private (self);

  g_clear_object (&priv->plugin_info);

  G_OBJECT_CLASS (foundry_dependency_provider_parent_class)->finalize (object);
}

static void
foundry_dependency_provider_get_property (GObject    *object,
                                          guint       prop_id,
                                          GValue     *value,
                                          GParamSpec *pspec)
{
  FoundryDependencyProvider *self = FOUNDRY_DEPENDENCY_PROVIDER (object);

  switch (prop_id)
    {
    case PROP_PLUGIN_INFO:
      g_value_take_object (value, foundry_dependency_provider_dup_plugin_info (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_dependency_provider_set_property (GObject      *object,
                                          guint         prop_id,
                                          const GValue *value,
                                          GParamSpec   *pspec)
{
  FoundryDependencyProvider *self = FOUNDRY_DEPENDENCY_PROVIDER (object);
  FoundryDependencyProviderPrivate *priv = foundry_dependency_provider_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_PLUGIN_INFO:
      priv->plugin_info = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_dependency_provider_class_init (FoundryDependencyProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_dependency_provider_finalize;
  object_class->get_property = foundry_dependency_provider_get_property;
  object_class->set_property = foundry_dependency_provider_set_property;

  klass->list_dependencies = foundry_dependency_provider_real_list_dependencies;

  properties[PROP_PLUGIN_INFO] =
    g_param_spec_object ("plugin-info", NULL, NULL,
                         PEAS_TYPE_PLUGIN_INFO,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_dependency_provider_init (FoundryDependencyProvider *self)
{
}

/**
 * foundry_dependency_provider_list_dependencies:
 * @self: a [class@Foundry.DependencyProvider]
 * @config: a [class@Foundry.Config]
 * @parent: (nullable): a [class@Foundry.Dependency] or %NULL
 *
 * Retrieves a list of dependencies for @config.
 *
 * @parent may be set to a previously provided [class@Foundry.Dependency] in
 * which case the provider must list the dependencies of that dependency or
 * reject with error if unsupported. `G_IO_ERROR_NOT_SUPPORTED` is the
 * preferred error code for this.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.Dependency] or rejects with
 *   an error.
 */
DexFuture *
foundry_dependency_provider_list_dependencies (FoundryDependencyProvider *self,
                                               FoundryConfig             *config,
                                               FoundryDependency         *parent)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEPENDENCY_PROVIDER (self));
  dex_return_error_if_fail (FOUNDRY_IS_CONFIG (config));
  dex_return_error_if_fail (!parent || FOUNDRY_IS_DEPENDENCY (parent));

  return FOUNDRY_DEPENDENCY_PROVIDER_GET_CLASS (self)->list_dependencies (self, config, parent);
}

/**
 * foundry_dependency_provider_dup_plugin_info:
 * @self: a [class@Foundry.DependencyProvider]
 *
 * Gets the plugin the provider belongs to.
 *
 * Returns: (transfer full): a [class@Peas.PluginInfo] or %NULL
 */
PeasPluginInfo *
foundry_dependency_provider_dup_plugin_info (FoundryDependencyProvider *self)
{
  FoundryDependencyProviderPrivate *priv = foundry_dependency_provider_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_DEPENDENCY_PROVIDER (self), NULL);

  return priv->plugin_info ? g_object_ref (priv->plugin_info) : NULL;
}

DexFuture *
foundry_dependency_provider_load (FoundryDependencyProvider *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEPENDENCY_PROVIDER (self));

  if (FOUNDRY_DEPENDENCY_PROVIDER_GET_CLASS (self)->load)
    return FOUNDRY_DEPENDENCY_PROVIDER_GET_CLASS (self)->load (self);

  return dex_future_new_true ();
}

DexFuture *
foundry_dependency_provider_unload (FoundryDependencyProvider *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEPENDENCY_PROVIDER (self));

  if (FOUNDRY_DEPENDENCY_PROVIDER_GET_CLASS (self)->unload)
    return FOUNDRY_DEPENDENCY_PROVIDER_GET_CLASS (self)->unload (self);

  return dex_future_new_true ();
}

/**
 * foundry_dependency_provider_update_dependencies:
 * @self: a [class@Foundry.DependencyProvider]
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_dependency_provider_update_dependencies (FoundryDependencyProvider *self,
                                                 FoundryConfig             *config,
                                                 GListModel                *dependencies,
                                                 int                        pty_fd,
                                                 DexCancellable            *cancellable)
{
  g_autoptr(DexCancellable) local_cancellable = NULL;
  g_autoptr(GListStore) filtered = NULL;
  guint n_items;

  dex_return_error_if_fail (FOUNDRY_IS_DEPENDENCY_PROVIDER (self));
  dex_return_error_if_fail (FOUNDRY_IS_CONFIG (config));
  dex_return_error_if_fail (G_IS_LIST_MODEL (dependencies));
  dex_return_error_if_fail (pty_fd >= -1);
  dex_return_error_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (!FOUNDRY_DEPENDENCY_PROVIDER_GET_CLASS (self)->update_dependencies)
    return dex_future_new_true ();

  if (cancellable == NULL)
    cancellable = local_cancellable = dex_cancellable_new ();

  filtered = g_list_store_new (FOUNDRY_TYPE_DEPENDENCY);
  n_items = g_list_model_get_n_items (dependencies);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryDependency) dependency = g_list_model_get_item (dependencies, i);
      g_autoptr(FoundryDependencyProvider) provider = foundry_dependency_dup_provider (dependency);

      if (provider != self)
        continue;

      g_list_store_append (filtered, dependency);
    }

  return FOUNDRY_DEPENDENCY_PROVIDER_GET_CLASS (self)->update_dependencies (self, config, G_LIST_MODEL (filtered), pty_fd, cancellable);
}
