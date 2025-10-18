/*
 * mks-mouse.h
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

#include <gio/gio.h>

#include "mks-types.h"
#include "mks-version-macros.h"

G_BEGIN_DECLS

#define MKS_TYPE_MOUSE            (mks_mouse_get_type ())
#define MKS_MOUSE(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), MKS_TYPE_MOUSE, MksMouse))
#define MKS_MOUSE_CONST(obj)      (G_TYPE_CHECK_INSTANCE_CAST ((obj), MKS_TYPE_MOUSE, MksMouse const))
#define MKS_MOUSE_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass),  MKS_TYPE_MOUSE, MksMouseClass))
#define MKS_IS_MOUSE(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), MKS_TYPE_MOUSE))
#define MKS_IS_MOUSE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass),  MKS_TYPE_MOUSE))
#define MKS_MOUSE_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj),  MKS_TYPE_MOUSE, MksMouseClass))

typedef struct _MksMouseClass MksMouseClass;

/**
 * MksMouseButton:
 * @MKS_MOUSE_BUTTON_LEFT: Left button.
 * @MKS_MOUSE_BUTTON_MIDDLE: Middle button.
 * @MKS_MOUSE_BUTTON_RIGHT: Right button.
 * @MKS_MOUSE_BUTTON_WHEEL_UP: Wheel-up button.
 * @MKS_MOUSE_BUTTON_WHEEL_DOWN: Wheel-down button.
 * @MKS_MOUSE_BUTTON_SIDE: Side button.
 * @MKS_MOUSE_BUTTON_EXTRA: Extra button.
 * 
 * A mouse button.
 */
typedef enum _MksMouseButton
{
  MKS_MOUSE_BUTTON_LEFT       = 0,
  MKS_MOUSE_BUTTON_MIDDLE     = 1,
  MKS_MOUSE_BUTTON_RIGHT      = 2,
  MKS_MOUSE_BUTTON_WHEEL_UP   = 3,
  MKS_MOUSE_BUTTON_WHEEL_DOWN = 4,
  MKS_MOUSE_BUTTON_SIDE       = 5,
  MKS_MOUSE_BUTTON_EXTRA      = 6,
} MksMouseButton;

MKS_AVAILABLE_IN_ALL
GType    mks_mouse_get_type        (void) G_GNUC_CONST;
MKS_AVAILABLE_IN_ALL
gboolean mks_mouse_get_is_absolute (MksMouse             *self);
MKS_AVAILABLE_IN_ALL
void     mks_mouse_press           (MksMouse             *self,
                                    MksMouseButton        button,
                                    GCancellable         *cancellable,
                                    GAsyncReadyCallback   callback,
                                    gpointer              user_data);
MKS_AVAILABLE_IN_ALL
gboolean mks_mouse_press_finish    (MksMouse             *self,
                                    GAsyncResult         *result,
                                    GError              **error);
MKS_AVAILABLE_IN_ALL
gboolean mks_mouse_press_sync      (MksMouse             *self,
                                    MksMouseButton        button,
                                    GCancellable         *cancellable,
                                    GError              **error);
MKS_AVAILABLE_IN_ALL
void     mks_mouse_release         (MksMouse             *self,
                                    MksMouseButton        button,
                                    GCancellable         *cancellable,
                                    GAsyncReadyCallback   callback,
                                    gpointer              user_data);
MKS_AVAILABLE_IN_ALL
gboolean mks_mouse_release_finish  (MksMouse             *self,
                                    GAsyncResult         *result,
                                    GError              **error);
MKS_AVAILABLE_IN_ALL
gboolean mks_mouse_release_sync    (MksMouse             *self,
                                    MksMouseButton        button,
                                    GCancellable         *cancellable,
                                    GError              **error);
MKS_AVAILABLE_IN_ALL
void     mks_mouse_move_to         (MksMouse             *self,
                                    guint                 x,
                                    guint                 y,
                                    GCancellable         *cancellable,
                                    GAsyncReadyCallback   callback,
                                    gpointer              user_data);
MKS_AVAILABLE_IN_ALL
gboolean mks_mouse_move_to_finish  (MksMouse             *self,
                                    GAsyncResult         *result,
                                    GError              **error);
MKS_AVAILABLE_IN_ALL
gboolean mks_mouse_move_to_sync    (MksMouse             *self,
                                    guint                 x,
                                    guint                 y,
                                    GCancellable         *cancellable,
                                    GError              **error);
MKS_AVAILABLE_IN_ALL
void     mks_mouse_move_by         (MksMouse             *self,
                                    int                   delta_x,
                                    int                   delta_y,
                                    GCancellable         *cancellable,
                                    GAsyncReadyCallback   callback,
                                    gpointer              user_data);
MKS_AVAILABLE_IN_ALL
gboolean mks_mouse_move_by_finish  (MksMouse             *self,
                                    GAsyncResult         *result,
                                    GError              **error);
MKS_AVAILABLE_IN_ALL
gboolean mks_mouse_move_by_sync    (MksMouse             *self,
                                    int                   delta_x,
                                    int                   delta_y,
                                    GCancellable         *cancellable,
                                    GError              **error);

G_END_DECLS
