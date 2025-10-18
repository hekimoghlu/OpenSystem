/*
 * mks-keyboard.h
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

#define MKS_TYPE_KEYBOARD            (mks_keyboard_get_type ())
#define MKS_KEYBOARD(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), MKS_TYPE_KEYBOARD, MksKeyboard))
#define MKS_KEYBOARD_CONST(obj)      (G_TYPE_CHECK_INSTANCE_CAST ((obj), MKS_TYPE_KEYBOARD, MksKeyboard const))
#define MKS_KEYBOARD_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass),  MKS_TYPE_KEYBOARD, MksKeyboardClass))
#define MKS_IS_KEYBOARD(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), MKS_TYPE_KEYBOARD))
#define MKS_IS_KEYBOARD_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass),  MKS_TYPE_KEYBOARD))
#define MKS_KEYBOARD_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj),  MKS_TYPE_KEYBOARD, MksKeyboardClass))

typedef struct _MksKeyboardClass MksKeyboardClass;

/**
 * MksKeyboardModifier:
 * @MKS_KEYBOARD_MODIFIER_NONE: No modifier.
 * @MKS_KEYBOARD_MODIFIER_SCROLL_LOCK: Scroll lock.
 * @MKS_KEYBOARD_MODIFIER_NUM_LOCK: Numeric lock.
 * @MKS_KEYBOARD_MODIFIER_CAPS_LOCK: Caps lock.
 * 
 * The active keyboard modifiers.
 */
typedef enum _MksKeyboardModifier
{
  MKS_KEYBOARD_MODIFIER_NONE        = 0,
  MKS_KEYBOARD_MODIFIER_SCROLL_LOCK = 1 << 0,
  MKS_KEYBOARD_MODIFIER_NUM_LOCK    = 1 << 1,
  MKS_KEYBOARD_MODIFIER_CAPS_LOCK   = 1 << 2,
} MksKeyboardModifier;

MKS_AVAILABLE_IN_ALL
GType               mks_keyboard_get_type       (void) G_GNUC_CONST;
MKS_AVAILABLE_IN_ALL
MksKeyboardModifier mks_keyboard_get_modifiers  (MksKeyboard          *self);
MKS_AVAILABLE_IN_ALL
void                mks_keyboard_press          (MksKeyboard          *self,
                                                 guint                 keycode,
                                                 GCancellable         *cancellable,
                                                 GAsyncReadyCallback   callback,
                                                 gpointer              user_data);
MKS_AVAILABLE_IN_ALL
gboolean            mks_keyboard_press_finish   (MksKeyboard          *self,
                                                 GAsyncResult         *result,
                                                 GError              **error);
MKS_AVAILABLE_IN_ALL
gboolean            mks_keyboard_press_sync     (MksKeyboard          *self,
                                                 guint                 keycode,
                                                 GCancellable         *cancellable,
                                                 GError              **error);
MKS_AVAILABLE_IN_ALL
void                mks_keyboard_release        (MksKeyboard          *self,
                                                 guint                 keycode,
                                                 GCancellable         *cancellable,
                                                 GAsyncReadyCallback   callback,
                                                 gpointer              user_data);
MKS_AVAILABLE_IN_ALL
gboolean            mks_keyboard_release_finish (MksKeyboard          *self,
                                                 GAsyncResult         *result,
                                                 GError              **error);
MKS_AVAILABLE_IN_ALL
gboolean            mks_keyboard_release_sync   (MksKeyboard          *self,
                                                 guint                 keycode,
                                                 GCancellable         *cancellable,
                                                 GError              **error);

MKS_AVAILABLE_IN_ALL
void                mks_keyboard_translate      (guint              keyval,
                                                 guint              keycode,
                                                 guint             *translated);

G_END_DECLS
