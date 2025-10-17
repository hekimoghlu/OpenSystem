/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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
#pragma once

namespace WebKit {

enum class WebEventType : uint8_t {
    // WebMouseEvent
    MouseDown,
    MouseUp,
    MouseMove,
    MouseForceChanged,
    MouseForceDown,
    MouseForceUp,

    // WebWheelEvent
    Wheel,

    // WebKeyboardEvent
    KeyDown,
    KeyUp,
    RawKeyDown,
    Char,

#if ENABLE(TOUCH_EVENTS)
    // WebTouchEvent
    TouchStart,
    TouchMove,
    TouchEnd,
    TouchCancel,
#endif

#if ENABLE(MAC_GESTURE_EVENTS)
    GestureStart,
    GestureChange,
    GestureEnd,
#endif
};

} // namespace WebKit
