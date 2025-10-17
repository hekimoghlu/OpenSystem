/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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

#include "WebMouseEvent.h"
#include <WebCore/PointerID.h>

#if USE(APPKIT)
#include <wtf/RetainPtr.h>
OBJC_CLASS NSView;
OBJC_CLASS NSEvent;
#endif

#if PLATFORM(GTK)
#include <WebCore/GRefPtrGtk.h>
#include <WebCore/GUniquePtrGtk.h>
#if USE(GTK4)
typedef struct _GdkEvent GdkEvent;
#else
typedef union _GdkEvent GdkEvent;
#endif
#endif

#if PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
typedef struct _WPEEvent WPEEvent;
#endif

#if PLATFORM(IOS_FAMILY)
#include <wtf/RetainPtr.h>
OBJC_CLASS WebEvent;
#endif

#if USE(LIBWPE)
struct wpe_input_pointer_event;
#endif

#if PLATFORM(WIN)
#include <windows.h>
#endif


namespace WebKit {

class NativeWebMouseEvent : public WebMouseEvent {
public:
#if USE(APPKIT)
    NativeWebMouseEvent(NSEvent *, NSEvent *lastPressureEvent, NSView *);
#elif PLATFORM(GTK)
    NativeWebMouseEvent(const NativeWebMouseEvent&);
    NativeWebMouseEvent(GdkEvent*, int, std::optional<WebCore::FloatSize>);
    NativeWebMouseEvent(GdkEvent*, const WebCore::IntPoint&, int, std::optional<WebCore::FloatSize>);
    NativeWebMouseEvent(WebEventType, WebMouseEventButton, unsigned short buttons, const WebCore::IntPoint& position, const WebCore::IntPoint& globalPosition, int clickCount, OptionSet<WebEventModifier> modifiers, std::optional<WebCore::FloatSize>, WebCore::PointerID, const String& pointerType, WebCore::PlatformMouseEvent::IsTouch isTouchEvent);
    explicit NativeWebMouseEvent(const WebCore::IntPoint&);
#elif PLATFORM(IOS_FAMILY)
    NativeWebMouseEvent(::WebEvent *);
    NativeWebMouseEvent(WebEventType, WebMouseEventButton, unsigned short buttons, const WebCore::IntPoint& position, const WebCore::IntPoint& globalPosition, float deltaX, float deltaY, float deltaZ, int clickCount, OptionSet<WebEventModifier>, WallTime timestamp, double force, GestureWasCancelled, const String& pointerType);
    NativeWebMouseEvent(const NativeWebMouseEvent&, const WebCore::IntPoint& position, const WebCore::IntPoint& globalPosition, float deltaX, float deltaY, float deltaZ);
#elif USE(LIBWPE)
    NativeWebMouseEvent(struct wpe_input_pointer_event*, float deviceScaleFactor, WebMouseEventSyntheticClickType = WebMouseEventSyntheticClickType::NoTap);
#if PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
    explicit NativeWebMouseEvent(WPEEvent*);
#endif
#elif PLATFORM(WIN)
    NativeWebMouseEvent(HWND, UINT message, WPARAM, LPARAM, bool, float deviceScaleFactor);
#endif

#if USE(APPKIT)
    NSEvent* nativeEvent() const { return m_nativeEvent.get(); }
#elif PLATFORM(GTK)
    const GdkEvent* nativeEvent() const { return m_nativeEvent.get(); }
#elif PLATFORM(IOS_FAMILY)
    ::WebEvent* nativeEvent() const { return m_nativeEvent.get(); }
#elif PLATFORM(WIN)
    const MSG* nativeEvent() const { return &m_nativeEvent; }
#else
    const void* nativeEvent() const { return nullptr; }
#endif

private:
#if USE(APPKIT)
    RetainPtr<NSEvent> m_nativeEvent;
#elif PLATFORM(GTK) && USE(GTK4)
    GRefPtr<GdkEvent> m_nativeEvent;
#elif PLATFORM(GTK)
    GUniquePtr<GdkEvent> m_nativeEvent;
#elif PLATFORM(IOS_FAMILY)
    RetainPtr<::WebEvent> m_nativeEvent;
#elif PLATFORM(WIN)
    MSG m_nativeEvent;
#endif
};

} // namespace WebKit
