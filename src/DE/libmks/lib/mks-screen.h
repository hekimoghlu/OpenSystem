/*
 * mks-screen.h
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

#include <gdk/gdk.h>
#include <gio/gio.h>

#include "mks-types.h"
#include "mks-version-macros.h"

G_BEGIN_DECLS

#define MKS_TYPE_SCREEN            (mks_screen_get_type ())
#define MKS_SCREEN(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), MKS_TYPE_SCREEN, MksScreen))
#define MKS_SCREEN_CONST(obj)      (G_TYPE_CHECK_INSTANCE_CAST ((obj), MKS_TYPE_SCREEN, MksScreen const))
#define MKS_SCREEN_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass),  MKS_TYPE_SCREEN, MksScreenClass))
#define MKS_IS_SCREEN(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), MKS_TYPE_SCREEN))
#define MKS_IS_SCREEN_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass),  MKS_TYPE_SCREEN))
#define MKS_SCREEN_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj),  MKS_TYPE_SCREEN, MksScreenClass))

typedef struct _MksScreenClass MksScreenClass;

/**
 * MksScreenKind:
 * @MKS_SCREEN_KIND_TEXT: A text only screen.
 * @MKS_SCREEN_KIND_GRAPHIC: A graphical screen.
 * 
 * A screen kind.
 */
typedef enum _MksScreenKind
{
  MKS_SCREEN_KIND_TEXT = 0,
  MKS_SCREEN_KIND_GRAPHIC = 1,
} MksScreenKind;

MKS_AVAILABLE_IN_ALL
GType          mks_screen_get_type           (void) G_GNUC_CONST;
MKS_AVAILABLE_IN_ALL
MksScreenKind  mks_screen_get_kind           (MksScreen            *self);
MKS_AVAILABLE_IN_ALL
MksKeyboard   *mks_screen_get_keyboard       (MksScreen            *self);
MKS_AVAILABLE_IN_ALL
MksMouse      *mks_screen_get_mouse          (MksScreen            *self);
MKS_AVAILABLE_IN_ALL
MksTouchable  *mks_screen_get_touchable      (MksScreen            *self);
MKS_AVAILABLE_IN_ALL
guint          mks_screen_get_width          (MksScreen            *self);
MKS_AVAILABLE_IN_ALL
guint          mks_screen_get_height         (MksScreen            *self);
MKS_AVAILABLE_IN_ALL
guint          mks_screen_get_number         (MksScreen            *self);
MKS_AVAILABLE_IN_ALL
const char    *mks_screen_get_device_address (MksScreen            *self);
MKS_AVAILABLE_IN_ALL
void           mks_screen_configure          (MksScreen            *self,
                                              MksScreenAttributes  *attributes,
                                              GCancellable         *cancellable,
                                              GAsyncReadyCallback   callback,
                                              gpointer              user_data);
MKS_AVAILABLE_IN_ALL
gboolean       mks_screen_configure_finish   (MksScreen            *self,
                                              GAsyncResult         *result,
                                              GError              **error);
MKS_AVAILABLE_IN_ALL
gboolean       mks_screen_configure_sync     (MksScreen            *self,
                                              MksScreenAttributes  *attributes,
                                              GCancellable         *cancellable,
                                              GError              **error);
MKS_AVAILABLE_IN_ALL
void           mks_screen_attach             (MksScreen            *self,
                                              GdkDisplay           *display,
                                              GCancellable         *cancellable,
                                              GAsyncReadyCallback   callback,
                                              gpointer              user_data);
MKS_AVAILABLE_IN_ALL
GdkPaintable  *mks_screen_attach_finish      (MksScreen            *self,
                                              GAsyncResult         *result,
                                              GError              **error);
MKS_AVAILABLE_IN_ALL
GdkPaintable  *mks_screen_attach_sync        (MksScreen            *self,
                                              GdkDisplay           *display,
                                              GCancellable         *cancellable,
                                              GError              **error);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (MksScreen, g_object_unref)

G_END_DECLS
