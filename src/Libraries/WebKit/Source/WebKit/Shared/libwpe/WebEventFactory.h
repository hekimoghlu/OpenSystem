/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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

#include "WebKeyboardEvent.h"
#include "WebMouseEvent.h"
#include "WebWheelEvent.h"

#if ENABLE(TOUCH_EVENTS)
#include "WebTouchEvent.h"
#endif

struct wpe_input_axis_event;
struct wpe_input_keyboard_event;
struct wpe_input_pointer_event;
#if ENABLE(TOUCH_EVENTS)
struct wpe_input_touch_event;
#endif

#if PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
typedef struct _WPEEvent WPEEvent;
#endif

namespace WebKit {

class WebEventFactory {
public:
    static WebKeyboardEvent createWebKeyboardEvent(struct wpe_input_keyboard_event*, const String&, bool isAutoRepeat, bool handledByInputMethod, std::optional<Vector<WebCore::CompositionUnderline>>&&, std::optional<EditingRange>&&);
    static WebMouseEvent createWebMouseEvent(struct wpe_input_pointer_event*, float deviceScaleFactor, WebMouseEventSyntheticClickType = WebMouseEventSyntheticClickType::NoTap);
    static WebWheelEvent createWebWheelEvent(struct wpe_input_axis_event*, float deviceScaleFactor, WebWheelEvent::Phase, WebWheelEvent::Phase momentumPhase);
#if ENABLE(TOUCH_EVENTS)
    static WebTouchEvent createWebTouchEvent(struct wpe_input_touch_event*, float deviceScaleFactor);
#endif

#if PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
    static WebMouseEvent createWebMouseEvent(WPEEvent*);
    static WebWheelEvent createWebWheelEvent(WPEEvent*);
    static WebWheelEvent createWebWheelEvent(WPEEvent*, WebWheelEvent::Phase);
    static WebKeyboardEvent createWebKeyboardEvent(WPEEvent*, const String&, bool isAutoRepeat);
#if ENABLE(TOUCH_EVENTS)
    static WebTouchEvent createWebTouchEvent(WPEEvent*, Vector<WebPlatformTouchPoint>&&);
#endif
#endif
};

} // namespace WebKit
