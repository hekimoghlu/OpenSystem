/*
 * mks-screen-attributes.h
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

#define MKS_TYPE_SCREEN_ATTRIBUTES (mks_screen_attributes_get_type ())

MKS_AVAILABLE_IN_ALL
GType                mks_screen_attributes_get_type      (void) G_GNUC_CONST;
MKS_AVAILABLE_IN_ALL
MksScreenAttributes *mks_screen_attributes_new           (void);
MKS_AVAILABLE_IN_ALL
MksScreenAttributes *mks_screen_attributes_copy          (MksScreenAttributes *self);
MKS_AVAILABLE_IN_ALL
void                 mks_screen_attributes_free          (MksScreenAttributes *self);
MKS_AVAILABLE_IN_ALL
gboolean             mks_screen_attributes_equal         (MksScreenAttributes *self,
                                                          MksScreenAttributes *other);
MKS_AVAILABLE_IN_ALL
void                 mks_screen_attributes_set_width_mm  (MksScreenAttributes *self,
                                                          guint16              width_mm);
MKS_AVAILABLE_IN_ALL
void                 mks_screen_attributes_set_height_mm (MksScreenAttributes *self,
                                                          guint16              height_mm);
MKS_AVAILABLE_IN_ALL
void                 mks_screen_attributes_set_x_offset  (MksScreenAttributes *self,
                                                          int                  x_offset);
MKS_AVAILABLE_IN_ALL
void                 mks_screen_attributes_set_y_offset  (MksScreenAttributes *self,
                                                          int                  y_offset);
MKS_AVAILABLE_IN_ALL
void                 mks_screen_attributes_set_width     (MksScreenAttributes *self,
                                                          guint                width);
MKS_AVAILABLE_IN_ALL
void                 mks_screen_attributes_set_height    (MksScreenAttributes *self,
                                                          guint                height);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (MksScreenAttributes, mks_screen_attributes_free)

G_END_DECLS
