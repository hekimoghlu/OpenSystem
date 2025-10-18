/* foundry-device-chassis.c
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

G_DEFINE_ENUM_TYPE (FoundryDeviceChassis, foundry_device_chassis,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEVICE_CHASSIS_WORKSTATION, "workstation"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEVICE_CHASSIS_HANDSET, "handset"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEVICE_CHASSIS_TABLET, "tablet"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEVICE_CHASSIS_OTHER, "other"))
