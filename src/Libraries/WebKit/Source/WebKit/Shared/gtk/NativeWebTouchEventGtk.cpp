/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#include "NativeWebTouchEvent.h"

#if ENABLE(TOUCH_EVENTS)

#include <WebCore/GtkVersioning.h>
#include "WebEventFactory.h"
#include <gdk/gdk.h>

namespace WebKit {

#if USE(GTK4)
#define constructNativeEvent(event) event
#else
#define constructNativeEvent(event) gdk_event_copy(event)
#endif

NativeWebTouchEvent::NativeWebTouchEvent(GdkEvent* event, Vector<WebPlatformTouchPoint>&& touchPoints)
    : WebTouchEvent(WebEventFactory::createWebTouchEvent(event, WTFMove(touchPoints)))
    , m_nativeEvent(constructNativeEvent(event))
{
}

NativeWebTouchEvent::NativeWebTouchEvent(const NativeWebTouchEvent& event)
    : WebTouchEvent(WebEventFactory::createWebTouchEvent(event.nativeEvent(), Vector<WebPlatformTouchPoint>(event.touchPoints())))
    , m_nativeEvent(constructNativeEvent(const_cast<GdkEvent*>(event.nativeEvent())))
{
}

} // namespace WebKit

#undef constructNativeEvent

#endif // ENABLE(TOUCH_EVENTS)
