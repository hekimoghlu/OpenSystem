/*
 * mks-touchable.h
 *
 * Copyright 2023 Bilal Elmoussaoui <belmouss@redhat.com>
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

#define MKS_TYPE_TOUCHABLE            (mks_touchable_get_type ())
#define MKS_TOUCHABLE(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), MKS_TYPE_TOUCHABLE, MksTouchable))
#define MKS_TOUCHABLE_CONST(obj)      (G_TYPE_CHECK_INSTANCE_CAST ((obj), MKS_TYPE_TOUCHABLE, MksTouchable const))
#define MKS_TOUCHABLE_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass),  MKS_TYPE_TOUCHABLE, MksTouchableClass))
#define MKS_IS_TOUCHABLE(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), MKS_TYPE_TOUCHABLE))
#define MKS_IS_TOUCHABLE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass),  MKS_TYPE_TOUCHABLE))
#define MKS_TOUCHABLE_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj),  MKS_TYPE_TOUCHABLE, MksTouchableClass))

typedef struct _MksTouchableClass MksTouchableClass;

/**
 * MksTouchEventKind:
 * @MKS_TOUCH_EVENT_BEGIN: The touch event has just started.
 * @MKS_TOUCH_EVENT_UPDATE: The touch event has been updated.
 * @MKS_TOUCH_EVENT_END: The touch event has finished.
 * @MKS_TOUCH_EVENT_CANCEL: The touch event has been canceled.
 *
 * The type of a touch event.
 */
typedef enum _MksTouchEventKind
{
  MKS_TOUCH_EVENT_BEGIN     = 0,
  MKS_TOUCH_EVENT_UPDATE    = 1,
  MKS_TOUCH_EVENT_END       = 2,
  MKS_TOUCH_EVENT_CANCEL    = 3,
} MksTouchEventKind;

MKS_AVAILABLE_IN_ALL
GType               mks_touchable_get_type          (void) G_GNUC_CONST;
MKS_AVAILABLE_IN_ALL
void                mks_touchable_send_event        (MksTouchable       *self,
                                                     MksTouchEventKind   kind,
                                                     guint64             num_slot,
                                                     double              x,
                                                     double              y,
                                                     GCancellable       *cancellable,
                                                     GAsyncReadyCallback callback,
                                                     gpointer            user_data);
MKS_AVAILABLE_IN_ALL
gboolean            mks_touchable_send_event_finish (MksTouchable       *self,
                                                     GAsyncResult       *result,
                                                     GError            **error);
MKS_AVAILABLE_IN_ALL
gboolean            mks_touchable_send_event_sync   (MksTouchable       *self,
                                                     MksTouchEventKind   kind,
                                                     guint64             num_slot,
                                                     double              x,
                                                     double              y,
                                                     GCancellable       *cancellable,
                                                     GError            **error);
MKS_AVAILABLE_IN_ALL
int                 mks_touchable_get_max_slots     (MksTouchable       *self);

G_END_DECLS
