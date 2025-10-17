/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 28, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef WPEViewQtQuick_h
#define WPEViewQtQuick_h

#include "WPEDisplayQtQuick.h"

class QHoverEvent;
class QKeyEvent;
class QMouseEvent;
class QSGTexture;
class QSize;
class QTouchEvent;
class QWheelEvent;
class WPEQtView;

G_BEGIN_DECLS

#define WPE_TYPE_VIEW_QTQUICK (wpe_view_qtquick_get_type())
G_DECLARE_FINAL_TYPE (WPEViewQtQuick, wpe_view_qtquick, WPE, VIEW_QTQUICK, WPEView)

WPEView *wpe_view_qtquick_new                              (WPEDisplayQtQuick *display);
gboolean         wpe_view_qtquick_initialize_rendering     (WPEViewQtQuick    *view, WPEQtView   *wpeQtView, GError **error);

QSGTexture*      wpe_view_qtquick_render_buffer_to_texture (WPEViewQtQuick    *view, QSize             size, GError **error);
void             wpe_view_qtquick_did_update_scene         (WPEViewQtQuick    *view);

void             wpe_view_dispatch_mouse_press_event       (WPEViewQtQuick    *view, QMouseEvent *);
void             wpe_view_dispatch_mouse_move_event        (WPEViewQtQuick    *view, QMouseEvent *);
void             wpe_view_dispatch_mouse_release_event     (WPEViewQtQuick    *view, QMouseEvent *);
void             wpe_view_dispatch_wheel_event             (WPEViewQtQuick    *view, QWheelEvent *);

void             wpe_view_dispatch_hover_enter_event       (WPEViewQtQuick    *view, QHoverEvent *);
void             wpe_view_dispatch_hover_move_event        (WPEViewQtQuick    *view, QHoverEvent *);
void             wpe_view_dispatch_hover_leave_event       (WPEViewQtQuick    *view, QHoverEvent *);

void             wpe_view_dispatch_key_press_event         (WPEViewQtQuick    *view, QKeyEvent *);
void             wpe_view_dispatch_key_release_event       (WPEViewQtQuick    *view, QKeyEvent *);

void             wpe_view_dispatch_touch_event             (WPEViewQtQuick    *view, QTouchEvent *);

G_END_DECLS

#endif /* WPEViewQtQuick_h */
