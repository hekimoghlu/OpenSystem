/* foundry-device.c
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

#include "foundry-device-chassis.h"
#include "foundry-device-manager.h"
#include "foundry-device-private.h"
#include "foundry-device-provider.h"
#include "foundry-triplet.h"

typedef struct _FoundryDevicePrivate
{
  GWeakRef provider_wr;
} FoundryDevicePrivate;

enum {
  PROP_0,
  PROP_ACTIVE,
  PROP_ID,
  PROP_PROVIDER,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryDevice, foundry_device, FOUNDRY_TYPE_CONTEXTUAL)

static GParamSpec *properties[N_PROPS];

static void
foundry_device_finalize (GObject *object)
{
  FoundryDevice *self = (FoundryDevice *)object;
  FoundryDevicePrivate *priv = foundry_device_get_instance_private (self);

  g_weak_ref_clear (&priv->provider_wr);

  G_OBJECT_CLASS (foundry_device_parent_class)->finalize (object);
}

static void
foundry_device_get_property (GObject    *object,
                             guint       prop_id,
                             GValue     *value,
                             GParamSpec *pspec)
{
  FoundryDevice *self = FOUNDRY_DEVICE (object);

  switch (prop_id)
    {
    case PROP_ACTIVE:
      g_value_set_boolean (value, foundry_device_get_active (self));
      break;

    case PROP_ID:
      g_value_take_string (value, foundry_device_dup_id (self));
      break;

    case PROP_PROVIDER:
      g_value_take_object (value, _foundry_device_dup_provider (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}


static void
foundry_device_class_init (FoundryDeviceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_device_finalize;
  object_class->get_property = foundry_device_get_property;

  properties[PROP_ACTIVE] =
    g_param_spec_boolean ("active", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PROVIDER] =
    g_param_spec_object ("provider", NULL, NULL,
                         FOUNDRY_TYPE_DEVICE_PROVIDER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_device_init (FoundryDevice *self)
{
  FoundryDevicePrivate *priv = foundry_device_get_instance_private (self);

  g_weak_ref_init (&priv->provider_wr, NULL);
}

/**
 * foundry_device_dup_id:
 * @self: a #FoundryDevice
 *
 * Gets the user-visible id for the device.
 *
 * Returns: (transfer full): a newly allocated string
 */
char *
foundry_device_dup_id (FoundryDevice *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEVICE (self), NULL);

  return FOUNDRY_DEVICE_GET_CLASS (self)->dup_id (self);
}

FoundryDeviceProvider *
_foundry_device_dup_provider (FoundryDevice *self)
{
  FoundryDevicePrivate *priv = foundry_device_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_DEVICE (self), NULL);

  return g_weak_ref_get (&priv->provider_wr);
}

void
_foundry_device_set_provider (FoundryDevice         *self,
                              FoundryDeviceProvider *provider)
{
  FoundryDevicePrivate *priv = foundry_device_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_DEVICE (self));
  g_return_if_fail (!provider || FOUNDRY_IS_DEVICE_PROVIDER (provider));

  g_weak_ref_set (&priv->provider_wr, provider);
}

gboolean
foundry_device_get_active (FoundryDevice *self)
{
  g_autoptr(FoundryContext) context = NULL;

  g_return_val_if_fail (FOUNDRY_IS_DEVICE (self), FALSE);

  if ((context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    {
      g_autoptr(FoundryDeviceManager) device_manager = foundry_context_dup_device_manager (context);
      g_autoptr(FoundryDevice) device = foundry_device_manager_dup_device (device_manager);

      return device == self;
    }

  return FALSE;
}

/**
 * foundry_device_load_info:
 * @self: a [class@Foundry.Device]
 *
 * Loads information about the device.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [class@Foundry.DeviceInfo].
 */
DexFuture *
foundry_device_load_info (FoundryDevice *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEVICE (self));

  return FOUNDRY_DEVICE_GET_CLASS (self)->load_info (self);
}
