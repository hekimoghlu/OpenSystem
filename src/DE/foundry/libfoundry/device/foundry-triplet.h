/* foundry-triplet.h
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

#pragma once

#include <glib-object.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TRIPLET (foundry_triplet_get_type())

FOUNDRY_AVAILABLE_IN_ALL
GType           foundry_triplet_get_type             (void);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTriplet *foundry_triplet_new                  (const char     *full_name);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTriplet *foundry_triplet_new_from_system      (void);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTriplet *foundry_triplet_new_with_triplet     (const char     *arch,
                                                      const char     *kernel,
                                                      const char     *operating_system);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTriplet *foundry_triplet_new_with_quadruplet  (const char     *arch,
                                                      const char     *vendor,
                                                      const char     *kernel,
                                                      const char     *operating_system);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTriplet *foundry_triplet_ref                  (FoundryTriplet *self);
FOUNDRY_AVAILABLE_IN_ALL
void            foundry_triplet_unref                (FoundryTriplet *self);
FOUNDRY_AVAILABLE_IN_ALL
const char     *foundry_triplet_get_full_name        (FoundryTriplet *self);
FOUNDRY_AVAILABLE_IN_ALL
const char     *foundry_triplet_get_arch             (FoundryTriplet *self);
FOUNDRY_AVAILABLE_IN_ALL
const char     *foundry_triplet_get_vendor           (FoundryTriplet *self);
FOUNDRY_AVAILABLE_IN_ALL
const char     *foundry_triplet_get_kernel           (FoundryTriplet *self);
FOUNDRY_AVAILABLE_IN_ALL
const char     *foundry_triplet_get_operating_system (FoundryTriplet *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean        foundry_triplet_is_system            (FoundryTriplet *self);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (FoundryTriplet, foundry_triplet_unref)

G_END_DECLS
