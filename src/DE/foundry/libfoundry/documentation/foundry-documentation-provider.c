/* foundry-documentation-provider.c
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
#include "foundry-documentation.h"
#include "foundry-documentation-matches.h"
#include "foundry-documentation-provider-private.h"
#include "foundry-documentation-query.h"
#include "foundry-documentation-root.h"

typedef struct
{
  PeasPluginInfo *plugin_info;
} FoundryDocumentationProviderPrivate;

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryDocumentationProvider, foundry_documentation_provider, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_PLUGIN_INFO,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_documentation_provider_finalize (GObject *object)
{
  FoundryDocumentationProvider *self = (FoundryDocumentationProvider *)object;
  FoundryDocumentationProviderPrivate *priv = foundry_documentation_provider_get_instance_private (self);

  g_clear_object (&priv->plugin_info);

  G_OBJECT_CLASS (foundry_documentation_provider_parent_class)->finalize (object);
}

static void
foundry_documentation_provider_get_property (GObject    *object,
                                             guint       prop_id,
                                             GValue     *value,
                                             GParamSpec *pspec)
{
  FoundryDocumentationProvider *self = FOUNDRY_DOCUMENTATION_PROVIDER (object);

  switch (prop_id)
    {
    case PROP_PLUGIN_INFO:
      g_value_take_object (value, foundry_documentation_provider_dup_plugin_info (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_documentation_provider_set_property (GObject      *object,
                                             guint         prop_id,
                                             const GValue *value,
                                             GParamSpec   *pspec)
{
  FoundryDocumentationProvider *self = FOUNDRY_DOCUMENTATION_PROVIDER (object);
  FoundryDocumentationProviderPrivate *priv = foundry_documentation_provider_get_instance_private (self);

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
foundry_documentation_provider_class_init (FoundryDocumentationProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_documentation_provider_finalize;
  object_class->get_property = foundry_documentation_provider_get_property;
  object_class->set_property = foundry_documentation_provider_set_property;

  properties[PROP_PLUGIN_INFO] =
    g_param_spec_object ("plugin-info", NULL, NULL,
                         PEAS_TYPE_PLUGIN_INFO,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_documentation_provider_init (FoundryDocumentationProvider *self)
{
}

/**
 * foundry_documentation_provider_dup_plugin_info:
 * @self: a [class@Foundry.DocumentationProvider]
 *
 * Gets the plugin the provider belongs to.
 *
 * Returns: (transfer full): a [class@Peas.PluginInfo] or %NULL
 */
PeasPluginInfo *
foundry_documentation_provider_dup_plugin_info (FoundryDocumentationProvider *self)
{
  FoundryDocumentationProviderPrivate *priv = foundry_documentation_provider_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_PROVIDER (self), NULL);

  return priv->plugin_info ? g_object_ref (priv->plugin_info) : NULL;
}

DexFuture *
foundry_documentation_provider_load (FoundryDocumentationProvider *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_PROVIDER (self));

  if (FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->load)
    return FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->load (self);

  return dex_future_new_true ();
}

DexFuture *
foundry_documentation_provider_unload (FoundryDocumentationProvider *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_PROVIDER (self));

  if (FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->unload)
    return FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->unload (self);

  return dex_future_new_true ();
}

/**
 * foundry_documentation_provider_list_roots:
 * @self: a [class@Foundry.DocumentationProvider]
 *
 * Returns a list of [class@Foundry.DocumentationRoot] that may contain
 * documentation to be discovered and ingested. This allows plugins for
 * SDKs to provide information about where documentation is located.
 *
 * It is expected that this list model will be updated when there are
 * changes to the underlying file-system which will require re-parsing
 * content for updates.
 *
 * Returns: (transfer full) (not nullable): a [iface@Gio.ListModel] of
 *   [class@Foundry.DocumentationRoot] containing information about
 *   discovering documentation.
 */
GListModel *
foundry_documentation_provider_list_roots (FoundryDocumentationProvider *self)
{
  GListModel *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_PROVIDER (self), NULL);

  if (FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->list_roots)
    ret = FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->list_roots (self);

  return ret ? ret : G_LIST_MODEL (g_list_store_new (FOUNDRY_TYPE_DOCUMENTATION_ROOT));
}

/**
 * foundry_documentation_provider_index:
 * @self: a [class@Foundry.DocumentationProvider]
 * @roots: a [iface@Gio.ListModel] of [class@Foundry.DocumentationRoot]
 *
 * This method is called when the documentation provider should rescan
 * the provided roots for changes.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value or rejects with error.
 */
DexFuture *
foundry_documentation_provider_index (FoundryDocumentationProvider *self,
                                      GListModel                   *roots)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_PROVIDER (self));
  dex_return_error_if_fail (G_IS_LIST_MODEL (roots));

  if (g_list_model_get_n_items (roots) == 0)
    return dex_future_new_true ();

  if (FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->index)
    return FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->index (self, roots);

  return dex_future_new_true ();
}

/**
 * foundry_documentation_provider_query:
 * @self: a [class@Foundry.DocumentationProvider]
 * @query: a [class@Foundry.DocumentationQuery]
 * @matches: a [class@Foundry.DocumentationMatches]
 *
 * Providers are expected to add their search sections to @matches
 * using [method@Foundry.DocumentationMatches.add_section].
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   any value or rejects with error.
 */
DexFuture *
foundry_documentation_provider_query (FoundryDocumentationProvider *self,
                                      FoundryDocumentationQuery    *query,
                                      FoundryDocumentationMatches  *matches)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_PROVIDER (self));
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_QUERY (query));
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_MATCHES (matches));

  if (FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->query)
    return FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->query (self, query, matches);

  return dex_future_new_true ();
}

/**
 * foundry_documentation_provider_list_children:
 * @self: a [class@Foundry.DocumentationProvider]
 *
 * Returns: (transfer full): a [class@Dex.Future] that will resolve to
 *   a [iface@Gio.ListModel] or reject with error.
 */
DexFuture *
foundry_documentation_provider_list_children (FoundryDocumentationProvider *self,
                                              FoundryDocumentation         *parent)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_PROVIDER (self));
  dex_return_error_if_fail (!parent || FOUNDRY_IS_DOCUMENTATION (parent));

  if (FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->list_children)
    return FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->list_children (self, parent);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Not supported");
}

/**
 * foundry_documentation_provider_find_by_uri:
 * @self: a [class@Foundry.DocumentationProvider]
 *
 * Returns: (transfer full): a [class@Dex.Future] that will resolve to
 *   a [class@Foundry.Documentation] or reject with error.
 */
DexFuture *
foundry_documentation_provider_find_by_uri (FoundryDocumentationProvider *self,
                                            const char                   *uri)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_PROVIDER (self));
  dex_return_error_if_fail (uri != NULL);

  if (FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->find_by_uri)
    return FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->find_by_uri (self, uri);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Not supported");
}

/**
 * foundry_documentation_provider_list_bundles:
 * @self: a [class@Foundry.DocumentationProvider]
 *
 * Returns: (transfer full): a [class@Dex.Future] that will resolve to
 *   a [iface@Gio.ListModel] of [class@Foundry.DocumentationBundle] or
 *   reject with error.
 */
DexFuture *
foundry_documentation_provider_list_bundles (FoundryDocumentationProvider *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_PROVIDER (self));

  if (FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->list_bundles)
    return FOUNDRY_DOCUMENTATION_PROVIDER_GET_CLASS (self)->list_bundles (self);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Not supported");
}
