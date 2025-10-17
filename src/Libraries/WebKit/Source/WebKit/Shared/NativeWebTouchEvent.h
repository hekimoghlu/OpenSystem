/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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

#include "WebTouchEvent.h"

#if PLATFORM(IOS_FAMILY) && defined(__OBJC__)
#include <UIKit/UIKit.h>
#endif

#if ENABLE(TOUCH_EVENTS)

#if PLATFORM(IOS_FAMILY)
#include "WKTouchEventsGestureRecognizerTypes.h"
#elif PLATFORM(GTK)
#include <WebCore/GRefPtrGtk.h>
#include <WebCore/GUniquePtrGtk.h>
#elif USE(LIBWPE)
#include <wpe/wpe.h>
#endif

#if PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
typedef struct _WPEEvent WPEEvent;
#endif

#endif // ENABLE(TOUCH_EVENTS)

namespace WebKit {

struct WKTouchEvent;

#if ENABLE(TOUCH_EVENTS)

class NativeWebTouchEvent : public WebTouchEvent {
public:
#if PLATFORM(IOS_FAMILY)
#if defined(__OBJC__)
    explicit NativeWebTouchEvent(const WKTouchEvent&, UIKeyModifierFlags);
#endif
#elif PLATFORM(GTK)
    NativeWebTouchEvent(GdkEvent*, Vector<WebPlatformTouchPoint>&&);
    NativeWebTouchEvent(const NativeWebTouchEvent&);
    const GdkEvent* nativeEvent() const { return m_nativeEvent.get(); }
#elif USE(LIBWPE)
    NativeWebTouchEvent(struct wpe_input_touch_event*, float deviceScaleFactor);
    const struct wpe_input_touch_event_raw* nativeFallbackTouchPoint() const { return &m_fallbackTouchPoint; }
#if PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
    NativeWebTouchEvent(WPEEvent*, Vector<WebPlatformTouchPoint>&&);
#endif
#elif PLATFORM(WIN)
    NativeWebTouchEvent();
#endif

private:
#if PLATFORM(IOS_FAMILY) && defined(__OBJC__)
    Vector<WebPlatformTouchPoint> extractWebTouchPoints(const WKTouchEvent&);
    Vector<WebTouchEvent> extractCoalescedWebTouchEvents(const WKTouchEvent&, UIKeyModifierFlags);
    Vector<WebTouchEvent> extractPredictedWebTouchEvents(const WKTouchEvent&, UIKeyModifierFlags);
#endif

#if PLATFORM(GTK) && USE(GTK4)
    GRefPtr<GdkEvent> m_nativeEvent;
#elif PLATFORM(GTK)
    GUniquePtr<GdkEvent> m_nativeEvent;
#elif USE(LIBWPE)
    struct wpe_input_touch_event_raw m_fallbackTouchPoint;
#endif
};

#endif // ENABLE(TOUCH_EVENTS)

#if PLATFORM(IOS_FAMILY) && defined(__OBJC__)
OptionSet<WebEventModifier> webEventModifierFlags(UIKeyModifierFlags);
#endif

} // namespace WebKit
