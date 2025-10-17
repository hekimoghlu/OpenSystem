/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
#ifndef WPEGestureController_h
#define WPEGestureController_h

#if !defined(__WPE_PLATFORM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/wpe-platform.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/WPEDefines.h>

G_BEGIN_DECLS

/**
 * WPEGesture:
 * @WPE_GESTURE_NONE: no gesture.
 * @WPE_GESTURE_TAP: tap gesture that has a corresponding position.
 * @WPE_GESTURE_DRAG: drag gesture that has a corresponding position and delta.
 */
typedef enum {
    WPE_GESTURE_NONE,

    WPE_GESTURE_TAP,
    WPE_GESTURE_DRAG,
} WPEGesture;

typedef struct _WPEEvent WPEEvent;

#define WPE_TYPE_GESTURE_CONTROLLER (wpe_gesture_controller_get_type())
WPE_API G_DECLARE_INTERFACE (WPEGestureController, wpe_gesture_controller, WPE, GESTURE_CONTROLLER, GObject)

struct _WPEGestureControllerInterface
{
    GTypeInterface parent_interface;

    void        (* handle_event)         (WPEGestureController *controller,
                                          WPEEvent             *event );
    void        (* cancel)               (WPEGestureController *controller);
    WPEGesture  (* get_gesture)          (WPEGestureController *controller);
    gboolean    (* get_gesture_position) (WPEGestureController *controller,
                                          double               *x,
                                          double               *y);
    gboolean    (* get_gesture_delta)    (WPEGestureController *controller,
                                          double               *x,
                                          double               *y);
    gboolean    (* is_drag_begin)        (WPEGestureController *controller);
};

WPE_API void        wpe_gesture_controller_handle_event         (WPEGestureController *controller,
                                                                 WPEEvent             *event);
WPE_API void        wpe_gesture_controller_cancel               (WPEGestureController *controller);
WPE_API WPEGesture  wpe_gesture_controller_get_gesture          (WPEGestureController *controller);
WPE_API gboolean    wpe_gesture_controller_get_gesture_position (WPEGestureController *controller,
                                                                 double               *x,
                                                                 double               *y);
WPE_API gboolean    wpe_gesture_controller_get_gesture_delta    (WPEGestureController *controller,
                                                                 double               *x,
                                                                 double               *y);
WPE_API gboolean    wpe_gesture_controller_is_drag_begin        (WPEGestureController *controller);

G_END_DECLS

#endif /* WPEGestureController_h */
