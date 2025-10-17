/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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

#include "WebWheelEvent.h"

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

#if USE(LIBWPE)
struct wpe_input_axis_event;
#endif

#if PLATFORM(WIN)
#include <windows.h>
#endif

namespace WebKit {

class NativeWebWheelEvent : public WebWheelEvent {
public:
#if USE(APPKIT)
    NativeWebWheelEvent(NSEvent *, NSView *);
#elif PLATFORM(GTK)
    NativeWebWheelEvent(const NativeWebWheelEvent&);
    NativeWebWheelEvent(GdkEvent*, const WebCore::IntPoint& position, const WebCore::IntPoint& globalPosition, const WebCore::FloatSize& delta, const WebCore::FloatSize& wheelTicks, WebWheelEvent::Phase, WebWheelEvent::Phase momentumPhase, bool hasPreciseDeltas = false);
#elif USE(LIBWPE)
    NativeWebWheelEvent(struct wpe_input_axis_event*, float deviceScaleFactor, WebWheelEvent::Phase, WebWheelEvent::Phase momentumPhase);
#if PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
    explicit NativeWebWheelEvent(WPEEvent*);
    NativeWebWheelEvent(WPEEvent*, WebWheelEvent::Phase);
#endif

#elif PLATFORM(WIN)
    NativeWebWheelEvent(HWND, UINT message, WPARAM, LPARAM, float deviceScaleFactor);
#endif

#if USE(APPKIT)
    NSEvent* nativeEvent() const { return m_nativeEvent.get(); }
#elif PLATFORM(GTK)
    GdkEvent* nativeEvent() const { return m_nativeEvent.get(); }
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
#elif PLATFORM(WIN)
    MSG m_nativeEvent;
#endif
};

} // namespace WebKit
