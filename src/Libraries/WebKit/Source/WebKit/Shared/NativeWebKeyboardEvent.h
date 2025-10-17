/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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

#if USE(APPKIT)
#include <wtf/RetainPtr.h>
OBJC_CLASS NSView;
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
struct wpe_input_keyboard_event;
#endif

#if PLATFORM(WIN)
#include <windows.h>
#endif

namespace WebKit {
struct EditingRange;

class NativeWebKeyboardEvent : public WebKeyboardEvent {
public:
#if USE(APPKIT)
    // FIXME: Share iOS's HandledByInputMethod enum here instead of passing a boolean.
    NativeWebKeyboardEvent(NSEvent *, bool handledByInputMethod, bool replacesSoftSpace, const Vector<WebCore::KeypressCommand>&);
#elif PLATFORM(GTK)
    NativeWebKeyboardEvent(const NativeWebKeyboardEvent&);
    NativeWebKeyboardEvent(GdkEvent*, const String&, bool isAutoRepeat, Vector<String>&& commands);
    NativeWebKeyboardEvent(const String&, std::optional<Vector<WebCore::CompositionUnderline>>&&, std::optional<EditingRange>&&);
    NativeWebKeyboardEvent(WebEventType, const String& text, const String& key, const String& code, const String& keyIdentifier, int windowsVirtualKeyCode, int nativeVirtualKeyCode, Vector<String>&& commands, bool isAutoRepeat, bool isKeypad, OptionSet<WebEventModifier>);
#elif PLATFORM(IOS_FAMILY)
    enum class HandledByInputMethod : bool { No, Yes };
    NativeWebKeyboardEvent(::WebEvent *, HandledByInputMethod);
#elif USE(LIBWPE)
    enum class HandledByInputMethod : bool { No, Yes };
    NativeWebKeyboardEvent(struct wpe_input_keyboard_event*, const String&, bool isAutoRepeat, HandledByInputMethod, std::optional<Vector<WebCore::CompositionUnderline>>&&, std::optional<EditingRange>&&);
#if PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
    NativeWebKeyboardEvent(WPEEvent*, const String&, bool isAutoRepeat);
    NativeWebKeyboardEvent(const String&, std::optional<Vector<WebCore::CompositionUnderline>>&&, std::optional<EditingRange>&&);
#endif
#elif PLATFORM(WIN)
    NativeWebKeyboardEvent(HWND, UINT message, WPARAM, LPARAM, Vector<MSG>&& pendingCharEvents);
#endif

#if USE(APPKIT)
    NSEvent *nativeEvent() const { return m_nativeEvent.get(); }
#elif PLATFORM(GTK)
    GdkEvent* nativeEvent() const { return m_nativeEvent.get(); }
#elif PLATFORM(IOS_FAMILY)
    ::WebEvent* nativeEvent() const { return m_nativeEvent.get(); }
#elif PLATFORM(WIN)
    const MSG* nativeEvent() const { return &m_nativeEvent; }
    const Vector<MSG>& pendingCharEvents() const { return m_pendingCharEvents; }
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
    Vector<MSG> m_pendingCharEvents;
#endif
};

} // namespace WebKit
