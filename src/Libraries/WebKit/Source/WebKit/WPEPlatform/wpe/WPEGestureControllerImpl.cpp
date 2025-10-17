/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
#include "config.h"
#include "WPEGestureControllerImpl.h"

#include "WPEGestureDetector.h"
#include <wtf/glib/WTFGType.h>

struct _WPEGestureControllerImplPrivate {
    WPE::GestureDetector detector;
};

static void wpe_gesture_controller_interface_init(WPEGestureControllerInterface*);

WEBKIT_DEFINE_FINAL_TYPE_WITH_CODE(
    WPEGestureControllerImpl, wpe_gesture_controller_impl, G_TYPE_OBJECT, GObject,
    G_IMPLEMENT_INTERFACE(WPE_TYPE_GESTURE_CONTROLLER, wpe_gesture_controller_interface_init))

static void wpeHandleEvent(WPEGestureController* controller, WPEEvent* event)
{
    WPE_GESTURE_CONTROLLER_IMPL(controller)->priv->detector.handleEvent(event);
}

static void wpeCancel(WPEGestureController* controller)
{
    WPE_GESTURE_CONTROLLER_IMPL(controller)->priv->detector.reset();
}

static WPEGesture wpeGetGesture(WPEGestureController* controller)
{
    return WPE_GESTURE_CONTROLLER_IMPL(controller)->priv->detector.gesture();
}

static gboolean wpeGetGesturePosition(WPEGestureController* controller, double* x, double* y)
{
    if (auto position = WPE_GESTURE_CONTROLLER_IMPL(controller)->priv->detector.position()) {
        *x = position->x;
        *y = position->y;
        return TRUE;
    }
    return FALSE;
}

static gboolean wpeGetGestureDelta(WPEGestureController* controller, double* x, double* y)
{
    if (auto delta = WPE_GESTURE_CONTROLLER_IMPL(controller)->priv->detector.delta()) {
        *x = delta->x;
        *y = delta->y;
        return TRUE;
    }
    return FALSE;
}

static gboolean wpeIsDragBegin(WPEGestureController* controller)
{
    return WPE_GESTURE_CONTROLLER_IMPL(controller)->priv->detector.dragBegin();
}

static void wpe_gesture_controller_impl_class_init(WPEGestureControllerImplClass*)
{
}

static void wpe_gesture_controller_interface_init(WPEGestureControllerInterface* interface)
{
    interface->handle_event = wpeHandleEvent;
    interface->cancel = wpeCancel;
    interface->get_gesture = wpeGetGesture;
    interface->get_gesture_position = wpeGetGesturePosition;
    interface->get_gesture_delta = wpeGetGestureDelta;
    interface->is_drag_begin = wpeIsDragBegin;
}

WPEGestureController* wpeGestureControllerImplNew()
{
    return WPE_GESTURE_CONTROLLER(g_object_new(WPE_TYPE_GESTURE_CONTROLLER_IMPL, nullptr));
}
