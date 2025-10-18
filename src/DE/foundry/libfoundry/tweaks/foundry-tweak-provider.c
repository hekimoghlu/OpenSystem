/* foundry-tweak-provider.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "foundry-tweak-provider-private.h"
#include "foundry-util.h"

typedef struct
{
  FoundryTweakTree *tree;
  GArray           *registrations;
} FoundryTweakProviderPrivate;

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryTweakProvider, foundry_tweak_provider, FOUNDRY_TYPE_CONTEXTUAL)

static void
foundry_tweak_provider_class_init (FoundryTweakProviderClass *klass)
{
}

static void
foundry_tweak_provider_init (FoundryTweakProvider *self)
{
}

DexFuture *
_foundry_tweak_provider_load (FoundryTweakProvider *self,
                              FoundryTweakTree     *tree)
{
  FoundryTweakProviderPrivate *priv = foundry_tweak_provider_get_instance_private (self);

  dex_return_error_if_fail (FOUNDRY_IS_TWEAK_PROVIDER (self));
  dex_return_error_if_fail (FOUNDRY_IS_TWEAK_TREE (tree));

  priv->registrations = g_array_new (FALSE, FALSE, sizeof (guint));
  priv->tree = g_object_ref (tree);

  if (FOUNDRY_TWEAK_PROVIDER_GET_CLASS (self)->load)
    return FOUNDRY_TWEAK_PROVIDER_GET_CLASS (self)->load (self);

  return dex_future_new_true ();
}

DexFuture *
_foundry_tweak_provider_unload (FoundryTweakProvider *self)
{
  FoundryTweakProviderPrivate *priv = foundry_tweak_provider_get_instance_private (self);
  DexFuture *ret;

  dex_return_error_if_fail (FOUNDRY_IS_TWEAK_PROVIDER (self));

  if (FOUNDRY_TWEAK_PROVIDER_GET_CLASS (self)->unload)
    ret = FOUNDRY_TWEAK_PROVIDER_GET_CLASS (self)->unload (self);
  else
    ret = dex_future_new_true ();

  for (guint i = 0; i < priv->registrations->len; i++)
    {
      guint reg = g_array_index (priv->registrations, guint, i);

      foundry_tweak_tree_unregister (priv->tree, reg);
    }

  g_clear_object (&priv->tree);

  return ret;
}

guint
foundry_tweak_provider_register (FoundryTweakProvider   *self,
                                 const char             *gettext_package,
                                 const char             *base_path,
                                 const FoundryTweakInfo *info,
                                 guint                   n_infos,
                                 const char * const     *environment)
{
  FoundryTweakProviderPrivate *priv = foundry_tweak_provider_get_instance_private (self);
  guint reg;

  g_return_val_if_fail (FOUNDRY_IS_TWEAK_PROVIDER (self), 0);
  g_return_val_if_fail (info != NULL || n_infos == 0, 0);

  if (n_infos == 0)
    return 0;

  reg = foundry_tweak_tree_register (priv->tree, gettext_package, base_path, info, n_infos, environment);
  g_array_append_val (priv->registrations, reg);
  return reg;
}

void
foundry_tweak_provider_unregister (FoundryTweakProvider *self,
                                   guint                 registration)
{
  FoundryTweakProviderPrivate *priv = foundry_tweak_provider_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_TWEAK_PROVIDER (self));
  g_return_if_fail (registration != 0);

  for (guint i = 0; i < priv->registrations->len; i++)
    {
      if (g_array_index (priv->registrations, guint, i) == registration)
        {
          g_array_remove_index_fast (priv->registrations, i);
          break;
        }
    }

  foundry_tweak_tree_unregister (priv->tree, registration);
}
