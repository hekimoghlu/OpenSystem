/*
 * mks-device.c
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "mks-device-private.h"

/**
 * MksDevice:
 * 
 * An abstraction of a virtualized QEMU device.
 */

G_DEFINE_TYPE (MksDevice, mks_device, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_NAME,
  N_PROPS
};

static GParamSpec *properties [N_PROPS];

static gboolean
mks_device_real_setup (MksDevice     *device,
                       MksQemuObject *object)
{
  g_assert (MKS_IS_DEVICE (device));
  g_assert (MKS_QEMU_IS_OBJECT (object));

  return TRUE;
}

static void
mks_device_dispose (GObject *object)
{
  MksDevice *self = (MksDevice *)object;

  g_clear_weak_pointer (&self->session);
  g_clear_pointer (&self->name, g_free);
  g_clear_object (&self->object);

  G_OBJECT_CLASS (mks_device_parent_class)->dispose (object);
}

static void
mks_device_get_property (GObject    *object,
                         guint       prop_id,
                         GValue     *value,
                         GParamSpec *pspec)
{
  MksDevice *self = MKS_DEVICE (object);

  switch (prop_id)
    {
    case PROP_NAME:
      g_value_set_string (value, mks_device_get_name (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
mks_device_class_init (MksDeviceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = mks_device_dispose;
  object_class->get_property = mks_device_get_property;

  klass->setup = mks_device_real_setup;

  /**
   * MksDevice:name:
   * 
   * The device name.
  */
  properties [PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
mks_device_init (MksDevice *self)
{
}

/**
 * mks_device_get_name:
 * @self: A `MksDevice`
 * 
 * Gets the device name. 
 */
const char *
mks_device_get_name (MksDevice *self)
{
  g_return_val_if_fail (MKS_IS_DEVICE (self), NULL);

  return self->name;
}

void
_mks_device_set_name (MksDevice  *self,
                      const char *name)
{
  g_return_if_fail (MKS_IS_DEVICE (self));

  if (g_set_str (&self->name, name))
    g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_NAME]);
}

gpointer
_mks_device_new (GType          device_type,
                 MksSession    *session,
                 MksQemuObject *object)
{
  g_autoptr(MksDevice) self = NULL;

  g_return_val_if_fail (g_type_is_a (device_type, MKS_TYPE_DEVICE), NULL);
  g_return_val_if_fail (device_type != MKS_TYPE_DEVICE, NULL);
  g_return_val_if_fail (MKS_IS_SESSION (session), NULL);
  g_return_val_if_fail (MKS_QEMU_IS_OBJECT (object), NULL);

  if (!(self = g_object_new (device_type, NULL)))
    return NULL;

  g_set_weak_pointer (&self->session, session);
  self->object = g_object_ref (object);

  if (!MKS_DEVICE_GET_CLASS (self)->setup (self, object))
    return NULL;

  return g_steal_pointer (&self);
}
