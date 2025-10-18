/*
 * mks-device.h
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

#pragma once

#if !defined(MKS_INSIDE) && !defined(MKS_COMPILATION)
# error "Only <libmks.h> can be included directly."
#endif

#include <glib-object.h>

#include "mks-types.h"
#include "mks-version-macros.h"

G_BEGIN_DECLS

#define MKS_TYPE_DEVICE            (mks_device_get_type ())
#define MKS_DEVICE(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), MKS_TYPE_DEVICE, MksDevice))
#define MKS_DEVICE_CONST(obj)      (G_TYPE_CHECK_INSTANCE_CAST ((obj), MKS_TYPE_DEVICE, MksDevice const))
#define MKS_DEVICE_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass),  MKS_TYPE_DEVICE, MksDeviceClass))
#define MKS_IS_DEVICE(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), MKS_TYPE_DEVICE))
#define MKS_IS_DEVICE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass),  MKS_TYPE_DEVICE))
#define MKS_DEVICE_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj),  MKS_TYPE_DEVICE, MksDeviceClass))

typedef struct _MksDeviceClass MksDeviceClass;

MKS_AVAILABLE_IN_ALL
GType       mks_device_get_type (void) G_GNUC_CONST;
MKS_AVAILABLE_IN_ALL
const char *mks_device_get_name (MksDevice *self);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (MksDevice, g_object_unref)

G_END_DECLS
