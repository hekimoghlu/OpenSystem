/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
#include "WPEGestureController.h"

/**
 * WPEGestureController:
 * @See_also: #WPEView
 *
 * A gesture controller.
 *
 * This interface enables implementing custom gesture detection algorithms.
 * Objects of classes implementing this interface can be supplied to
 * #WPEView so that they are being used instead of the default gesture detector.
 */

G_DEFINE_INTERFACE(WPEGestureController, wpe_gesture_controller, G_TYPE_OBJECT)

static void wpe_gesture_controller_default_init(WPEGestureControllerInterface*)
{
}

/**
 * wpe_gesture_controller_handle_event:
 * @controller: a #WPEGestureController
 * @event: a #WPEEvent
 *
 * Get the gesture detected by @controller if any was detected during processing of @event.
 */
void wpe_gesture_controller_handle_event(WPEGestureController* controller, WPEEvent* event)
{
    g_return_if_fail(controller);
    g_return_if_fail(event);

    auto* controllerInterface = WPE_GESTURE_CONTROLLER_GET_IFACE(controller);
    controllerInterface->handle_event(controller, event);
}

/**
 * wpe_gesture_controller_cancel:
 * @controller: a #WPEGestureController
 *
 * Cancels ongoing gesture detection if any.
 */
void wpe_gesture_controller_cancel(WPEGestureController* controller)
{
    auto* controllerInterface = WPE_GESTURE_CONTROLLER_GET_IFACE(controller);
    controllerInterface->cancel(controller);
}

/**
 * wpe_gesture_controller_get_gesture:
 * @controller: a #WPEGestureController
 *
 * Get currently detected gesture.
 *
 * Returns: a #WPEGesture
 */
WPEGesture wpe_gesture_controller_get_gesture(WPEGestureController* controller)
{
    g_return_val_if_fail(controller, WPE_GESTURE_NONE);

    auto* controllerInterface = WPE_GESTURE_CONTROLLER_GET_IFACE(controller);
    return controllerInterface->get_gesture(controller);
}

/**
 * wpe_gesture_controller_get_gesture_position:
 * @controller: a #WPEGestureController
 * @x: (out): location to store x coordinate
 * @y: (out): location to store y coordinate
 *
 * Get the position of currently detected gesture. If it doesn't have
 * a position, %FALSE is returned.
 *
 * Returns: %TRUE if position is returned in @x and @y,
 *    or %FALSE if currently detected gesture doesn't have a positon
 */
gboolean wpe_gesture_controller_get_gesture_position(WPEGestureController* controller, double* x, double* y)
{
    g_return_val_if_fail(controller, FALSE);
    g_return_val_if_fail(x && y, FALSE);

    auto* controllerInterface = WPE_GESTURE_CONTROLLER_GET_IFACE(controller);
    return controllerInterface->get_gesture_position(controller, x, y);
}

/**
 * wpe_gesture_controller_get_gesture_delta:
 * @controller: a #WPEGestureController
 * @x: (out): location to store delta on x axis
 * @y: (out): location to store delta on y axis
 *
 * Get the delta of currently detected gesture such as "drag" gesture.
 * If it doesn't have a delta, %FALSE is returned.
 *
 * Returns: %TRUE if delta is returned in @x and @y,
 *    or %FALSE if currently detected gesture doesn't have a delta
 */
gboolean wpe_gesture_controller_get_gesture_delta(WPEGestureController* controller, double* x, double* y)
{
    g_return_val_if_fail(controller, FALSE);
    g_return_val_if_fail(x && y, FALSE);

    auto* controllerInterface = WPE_GESTURE_CONTROLLER_GET_IFACE(controller);
    return controllerInterface->get_gesture_delta(controller, x, y);
}

/**
 * wpe_gesture_controller_is_drag_begin:
 * @controller: a #WPEGestureController
 *
 * Check if the current drag gesture is a beginning of the sequence being detected.
 *
 * Returns: %TRUE if current drag gesture is a beginning of the sequence being detected,
 *    or %FALSE otherwise
 */
gboolean wpe_gesture_controller_is_drag_begin(WPEGestureController* controller)
{
    g_return_val_if_fail(controller, FALSE);

    auto* controllerInterface = WPE_GESTURE_CONTROLLER_GET_IFACE(controller);
    return controllerInterface->is_drag_begin(controller);
}
