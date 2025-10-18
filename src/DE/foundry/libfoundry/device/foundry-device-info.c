/* foundry-device-info.c
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

#include "foundry-device.h"
#include "foundry-device-chassis.h"
#include "foundry-device-info.h"
#include "foundry-triplet.h"

typedef struct
{
  GWeakRef device_wr;
} FoundryDeviceInfoPrivate;

enum {
  PROP_0,
  PROP_ACTIVE,
  PROP_CHASSIS,
  PROP_DEVICE,
  PROP_ID,
  PROP_NAME,
  PROP_TRIPLET,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryDeviceInfo, foundry_device_info, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_device_info_finalize (GObject *object)
{
  FoundryDeviceInfo *self = (FoundryDeviceInfo *)object;
  FoundryDeviceInfoPrivate *priv = foundry_device_info_get_instance_private (self);

  g_weak_ref_clear (&priv->device_wr);

  G_OBJECT_CLASS (foundry_device_info_parent_class)->finalize (object);
}

static void
foundry_device_info_get_property (GObject    *object,
                                  guint       prop_id,
                                  GValue     *value,
                                  GParamSpec *pspec)
{
  FoundryDeviceInfo *self = FOUNDRY_DEVICE_INFO (object);
  FoundryDeviceInfoPrivate *priv = foundry_device_info_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_ACTIVE:
      g_value_set_boolean (value, foundry_device_info_get_active (self));
      break;

    case PROP_DEVICE:
      g_value_take_object (value, g_weak_ref_get (&priv->device_wr));
      break;

    case PROP_ID:
      g_value_take_string (value, foundry_device_info_dup_id (self));
      break;

    case PROP_NAME:
      g_value_take_string (value, foundry_device_info_dup_name (self));
      break;

    case PROP_TRIPLET:
      g_value_take_boxed (value, foundry_device_info_dup_triplet (self));
      break;

    case PROP_CHASSIS:
      g_value_set_enum (value, foundry_device_info_get_chassis (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_device_info_set_property (GObject      *object,
                                  guint         prop_id,
                                  const GValue *value,
                                  GParamSpec   *pspec)
{
  FoundryDeviceInfo *self = FOUNDRY_DEVICE_INFO (object);
  FoundryDeviceInfoPrivate *priv = foundry_device_info_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_DEVICE:
      g_weak_ref_set (&priv->device_wr, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_device_info_class_init (FoundryDeviceInfoClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_device_info_finalize;
  object_class->get_property = foundry_device_info_get_property;
  object_class->set_property = foundry_device_info_set_property;

  properties[PROP_ACTIVE] =
    g_param_spec_boolean ("active", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_DEVICE] =
    g_param_spec_object ("device", NULL, NULL,
                         FOUNDRY_TYPE_DEVICE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_CHASSIS] =
    g_param_spec_enum ("chassis", NULL, NULL,
                       FOUNDRY_TYPE_DEVICE_CHASSIS,
                       FOUNDRY_DEVICE_CHASSIS_OTHER,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_TRIPLET] =
    g_param_spec_boxed ("triplet", NULL, NULL,
                        FOUNDRY_TYPE_TRIPLET,
                        (G_PARAM_READABLE |
                         G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_device_info_init (FoundryDeviceInfo *self)
{
  FoundryDeviceInfoPrivate *priv = foundry_device_info_get_instance_private (self);

  g_weak_ref_init (&priv->device_wr, NULL);
}

char *
foundry_device_info_dup_id (FoundryDeviceInfo *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEVICE_INFO (self), NULL);

  return FOUNDRY_DEVICE_INFO_GET_CLASS (self)->dup_id (self);
}

char *
foundry_device_info_dup_name (FoundryDeviceInfo *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEVICE_INFO (self), NULL);

  return FOUNDRY_DEVICE_INFO_GET_CLASS (self)->dup_name (self);
}

FoundryDeviceChassis
foundry_device_info_get_chassis (FoundryDeviceInfo *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEVICE_INFO (self), 0);

  if (FOUNDRY_DEVICE_INFO_GET_CLASS (self)->get_chassis)
    return FOUNDRY_DEVICE_INFO_GET_CLASS (self)->get_chassis (self);

  return FOUNDRY_DEVICE_CHASSIS_OTHER;
}

/**
 * foundry_device_info_dup_triplet:
 * @self: a [class@Foundry.DeviceInfo]
 *
 * Returns: (transfer full): a #FoundryTriplet
 */
FoundryTriplet *
foundry_device_info_dup_triplet (FoundryDeviceInfo *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEVICE_INFO (self), NULL);

  return FOUNDRY_DEVICE_INFO_GET_CLASS (self)->dup_triplet (self);
}

gboolean
foundry_device_info_get_active (FoundryDeviceInfo *self)
{
  FoundryDeviceInfoPrivate *priv = foundry_device_info_get_instance_private (self);
  g_autoptr(FoundryDevice) device = NULL;

  g_return_val_if_fail (FOUNDRY_IS_DEVICE_INFO (self), FALSE);

  if ((device = g_weak_ref_get (&priv->device_wr)))
    return foundry_device_get_active (device);

  return FALSE;
}
