/*
 * mks-display-picture-private.h
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

#include <gtk/gtk.h>

#include "mks-paintable-private.h"
#include "mks-types.h"

G_BEGIN_DECLS

#define MKS_TYPE_DISPLAY_PICTURE (mks_display_picture_get_type())

G_DECLARE_FINAL_TYPE (MksDisplayPicture, mks_display_picture, MKS, DISPLAY_PICTURE, GtkWidget)

GtkWidget    *mks_display_picture_new           (void);
MksPaintable *mks_display_picture_get_paintable (MksDisplayPicture *self);
void          mks_display_picture_set_paintable (MksDisplayPicture *self,
                                                 MksPaintable      *paintable);
MksMouse     *mks_display_picture_get_mouse     (MksDisplayPicture *self);
void          mks_display_picture_set_mouse     (MksDisplayPicture *self,
                                                 MksMouse          *mouse);
MksKeyboard  *mks_display_picture_get_keyboard  (MksDisplayPicture *self);
void          mks_display_picture_set_keyboard  (MksDisplayPicture *self,
                                                 MksKeyboard       *keyboard);
MksTouchable *mks_display_picture_get_touchable (MksDisplayPicture *self);
void          mks_display_picture_set_touchable (MksDisplayPicture *self,
                                                 MksTouchable      *touchable);
gboolean
   mks_display_picture_event_get_guest_position (MksDisplayPicture *self,
                                                 GdkEvent          *event,
                                                 double            *guest_x,
                                                 double            *guest_y);
G_END_DECLS
