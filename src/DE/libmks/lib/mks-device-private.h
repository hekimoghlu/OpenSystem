/*
 * mks-device-private.h
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

#include "mks-device.h"
#include "mks-qemu.h"
#include "mks-session.h"

G_BEGIN_DECLS

struct _MksDevice
{
  GObject        parent_instance;
  MksSession    *session;
  MksQemuObject *object;
  char          *name;
};

struct _MksDeviceClass
{
  GObjectClass parent_class;

  gboolean (*setup) (MksDevice     *self,
                     MksQemuObject *object);
};

gpointer _mks_device_new      (GType          device_type,
                               MksSession    *session,
                               MksQemuObject *object);
void     _mks_device_set_name (MksDevice     *self,
                               const char    *name);

G_END_DECLS
