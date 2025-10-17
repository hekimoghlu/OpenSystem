/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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

#if USE(GTK4)
typedef struct _GdkEvent GdkEvent;
#else
typedef union _GdkEvent GdkEvent;
#endif

namespace WebKit {

class WebEventFactory {
public:
    static WebMouseEvent createWebMouseEvent(const GdkEvent*, int, std::optional<WebCore::FloatSize>);
    static WebMouseEvent createWebMouseEvent(const GdkEvent*, const WebCore::IntPoint&, const WebCore::IntPoint&, int, std::optional<WebCore::FloatSize>);
    static WebMouseEvent createWebMouseEvent(const WebCore::IntPoint&);
    static WebKeyboardEvent createWebKeyboardEvent(const GdkEvent*, const String&, bool isAutoRepeat, bool handledByInputMethod, std::optional<Vector<WebCore::CompositionUnderline>>&&, std::optional<EditingRange>&&, Vector<String>&& commands);
#if ENABLE(TOUCH_EVENTS)
    static WebTouchEvent createWebTouchEvent(const GdkEvent*, Vector<WebPlatformTouchPoint>&&);
#endif
    static WebWheelEvent createWebWheelEvent(const GdkEvent*, const WebCore::IntPoint&, const WebCore::IntPoint&, const WebCore::FloatSize&, const WebCore::FloatSize&, WebWheelEvent::Phase, WebWheelEvent::Phase, bool hasPreciseDeltas);
};

} // namespace WebKit
