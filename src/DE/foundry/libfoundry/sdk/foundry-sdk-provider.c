/* foundry-sdk-provider.c
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

#include <glib/gi18n-lib.h>

#include "foundry-sdk-provider-private.h"
#include "foundry-sdk-private.h"
#include "foundry-util.h"

typedef struct
{
  GPtrArray *sdks;
} FoundrySdkProviderPrivate;

static void list_model_iface_init (GListModelInterface *iface);

G_DEFINE_ABSTRACT_TYPE_WITH_CODE (FoundrySdkProvider, foundry_sdk_provider, FOUNDRY_TYPE_CONTEXTUAL,
                                  G_ADD_PRIVATE (FoundrySdkProvider)
                                  G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static DexFuture *
foundry_sdk_provider_real_load (FoundrySdkProvider *self)
{
  return dex_future_new_true ();
}

static DexFuture *
foundry_sdk_provider_real_unload (FoundrySdkProvider *self)
{
  FoundrySdkProviderPrivate *priv = foundry_sdk_provider_get_instance_private (self);
  guint n_items;

  g_assert (FOUNDRY_IS_SDK_PROVIDER (self));

  n_items = priv->sdks->len;

  if (n_items > 0)
    {
      g_ptr_array_remove_range (priv->sdks, 0, n_items);
      g_list_model_items_changed (G_LIST_MODEL (self), 0, n_items, 0);
    }

  return dex_future_new_true ();
}

static void
foundry_sdk_provider_finalize (GObject *object)
{
  FoundrySdkProvider *self = (FoundrySdkProvider *)object;
  FoundrySdkProviderPrivate *priv = foundry_sdk_provider_get_instance_private (self);

  g_clear_pointer (&priv->sdks, g_ptr_array_unref);

  G_OBJECT_CLASS (foundry_sdk_provider_parent_class)->finalize (object);
}

static void
foundry_sdk_provider_class_init (FoundrySdkProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_sdk_provider_finalize;

  klass->load = foundry_sdk_provider_real_load;
  klass->unload = foundry_sdk_provider_real_unload;
}

static void
foundry_sdk_provider_init (FoundrySdkProvider *self)
{
  FoundrySdkProviderPrivate *priv = foundry_sdk_provider_get_instance_private (self);

  priv->sdks = g_ptr_array_new_with_free_func (g_object_unref);
}

void
foundry_sdk_provider_sdk_added (FoundrySdkProvider *self,
                                FoundrySdk         *sdk)
{
  FoundrySdkProviderPrivate *priv = foundry_sdk_provider_get_instance_private (self);
  guint position;

  g_return_if_fail (FOUNDRY_IS_SDK_PROVIDER (self));
  g_return_if_fail (FOUNDRY_IS_SDK (sdk));

  _foundry_sdk_set_provider (sdk, self);

  position = priv->sdks->len;

  g_ptr_array_add (priv->sdks, g_object_ref (sdk));
  g_list_model_items_changed (G_LIST_MODEL (self), position, 0, 1);
}

void
foundry_sdk_provider_sdk_removed (FoundrySdkProvider *self,
                                  FoundrySdk         *sdk)
{
  FoundrySdkProviderPrivate *priv = foundry_sdk_provider_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_SDK_PROVIDER (self));
  g_return_if_fail (FOUNDRY_IS_SDK (sdk));

  for (guint i = 0; i < priv->sdks->len; i++)
    {
      FoundrySdk *element = g_ptr_array_index (priv->sdks, i);

      if (element == sdk)
        {
          _foundry_sdk_set_provider (sdk, NULL);
          g_ptr_array_remove_index (priv->sdks, i);
          g_list_model_items_changed (G_LIST_MODEL (self), i, 1, 0);
          return;
        }
    }

  g_critical ("%s did not contain sdk %s at %p",
              G_OBJECT_TYPE_NAME (self),
              G_OBJECT_TYPE_NAME (sdk),
              sdk);
}

static GType
foundry_sdk_provider_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_SDK;
}

static guint
foundry_sdk_provider_get_n_items (GListModel *model)
{
  FoundrySdkProvider *self = FOUNDRY_SDK_PROVIDER (model);
  FoundrySdkProviderPrivate *priv = foundry_sdk_provider_get_instance_private (self);

  return priv->sdks->len;
}

static gpointer
foundry_sdk_provider_get_item (GListModel *model,
                               guint       position)
{
  FoundrySdkProvider *self = FOUNDRY_SDK_PROVIDER (model);
  FoundrySdkProviderPrivate *priv = foundry_sdk_provider_get_instance_private (self);

  if (position < priv->sdks->len)
    return g_object_ref (g_ptr_array_index (priv->sdks, position));

  return NULL;
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_sdk_provider_get_item_type;
  iface->get_n_items = foundry_sdk_provider_get_n_items;
  iface->get_item = foundry_sdk_provider_get_item;
}

DexFuture *
foundry_sdk_provider_load (FoundrySdkProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SDK_PROVIDER (self), NULL);

  return FOUNDRY_SDK_PROVIDER_GET_CLASS (self)->load (self);
}

DexFuture *
foundry_sdk_provider_unload (FoundrySdkProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SDK_PROVIDER (self), NULL);

  return FOUNDRY_SDK_PROVIDER_GET_CLASS (self)->unload (self);
}

/**
 * foundry_sdk_provider_dup_name:
 * @self: a #FoundrySdkProvider
 *
 * Gets a name for the provider that is expected to be displayed to
 * users such as "Flatpak".
 *
 * Returns: (transfer full): the name of the provider
 */
char *
foundry_sdk_provider_dup_name (FoundrySdkProvider *self)
{
  char *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_SDK_PROVIDER (self), NULL);

  if (FOUNDRY_SDK_PROVIDER_GET_CLASS (self)->dup_name)
    ret = FOUNDRY_SDK_PROVIDER_GET_CLASS (self)->dup_name (self);

  if (ret == NULL)
    ret = g_strdup (G_OBJECT_TYPE_NAME (self));

 return g_steal_pointer (&ret);
}

static gboolean
equal_by_id (gconstpointer a,
             gconstpointer b)
{
  g_autofree char *a_id = foundry_sdk_dup_id ((FoundrySdk *)a);
  g_autofree char *b_id = foundry_sdk_dup_id ((FoundrySdk *)b);

  return g_strcmp0 (a_id, b_id) == 0;
}

/**
 * foundry_sdk_provider_merge:
 * @self: a #FoundrySdkProvider
 * @sdks: (element-type Foundry.Sdk): a #GPtrArray of SDKs
 *
 * This is a convenience function for SDK providers that need to
 * parse the whole set of SDKs when doing updating. Just provide
 * them all as a list here and only the changes will be applied.
 */
void
foundry_sdk_provider_merge (FoundrySdkProvider *self,
                            GPtrArray          *sdks)
{
  FoundrySdkProviderPrivate *priv = foundry_sdk_provider_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_SDK_PROVIDER (self));
  g_return_if_fail (sdks != NULL);

  /* First remove any SDKs not in the set, or replace them with
   * the new version of the object. Scan in reverse so that we can
   * have stable indexes.
   */
  for (guint i = priv->sdks->len; i > 0; i--)
    {
      FoundrySdk *sdk = g_ptr_array_index (priv->sdks, i-1);
      guint position;

      if (g_ptr_array_find_with_equal_func (sdks, sdk, equal_by_id, &position))
        {
          g_ptr_array_index (priv->sdks, i-1) = g_object_ref (g_ptr_array_index (sdks, position));
          g_list_model_items_changed (G_LIST_MODEL (self), i-1, 1, 1);
          continue;
        }

      foundry_sdk_provider_sdk_removed (self, sdk);
    }

  for (guint i = 0; i < sdks->len; i++)
    {
      FoundrySdk *sdk = g_ptr_array_index (sdks, i);
      guint position;

      if (!g_ptr_array_find_with_equal_func (priv->sdks, sdk, equal_by_id, &position))
        foundry_sdk_provider_sdk_added (self, sdk);
    }
}

/**
 * foundry_sdk_provider_find_by_id:
 * @self: a [class@Foundry.SdkProvider]
 * @sdk_id: the identifier for the SDK such as "org.gnome.Sdk/x86_64/master"
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.Sdk] or rejects with error.
 */
DexFuture *
foundry_sdk_provider_find_by_id (FoundrySdkProvider *self,
                                 const char         *sdk_id)
{
  FoundrySdkProviderPrivate *priv = foundry_sdk_provider_get_instance_private (self);

  dex_return_error_if_fail (FOUNDRY_IS_SDK_PROVIDER (self));
  dex_return_error_if_fail (sdk_id != NULL);

  for (guint i = 0; i < priv->sdks->len; i++)
    {
      FoundrySdk *sdk = g_ptr_array_index (priv->sdks, i);
      g_autofree char *id = foundry_sdk_dup_id (sdk);

      if (foundry_str_equal0 (id, sdk_id))
        return dex_future_new_take_object (g_object_ref (sdk));
    }

  if (FOUNDRY_SDK_PROVIDER_GET_CLASS (self)->find_by_id)
    return FOUNDRY_SDK_PROVIDER_GET_CLASS (self)->find_by_id (self, sdk_id);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}
