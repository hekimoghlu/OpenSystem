/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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

#include "EditingRange.h"
#include "WebEvent.h"
#include <WebCore/CompositionUnderline.h>

#if USE(APPKIT)
#include <WebCore/KeypressCommand.h>
#endif

namespace WebKit {

class WebKeyboardEvent : public WebEvent {
public:
    ~WebKeyboardEvent();

#if USE(APPKIT)
    WebKeyboardEvent(WebEvent&&, const String& text, const String& unmodifiedText, const String& key, const String& code, const String& keyIdentifier, int windowsVirtualKeyCode, int nativeVirtualKeyCode, int macCharCode, bool handledByInputMethod, const Vector<WebCore::KeypressCommand>&, bool isAutoRepeat, bool isKeypad, bool isSystemKey);
#elif PLATFORM(GTK)
    WebKeyboardEvent(WebEvent&&, const String& text, const String& key, const String& code, const String& keyIdentifier, int windowsVirtualKeyCode, int nativeVirtualKeyCode, bool handledByInputMethod, std::optional<Vector<WebCore::CompositionUnderline>>&&, std::optional<EditingRange>&&, Vector<String>&& commands, bool isAutoRepeat, bool isKeypad);
#elif PLATFORM(IOS_FAMILY)
    WebKeyboardEvent(WebEvent&&, const String& text, const String& unmodifiedText, const String& key, const String& code, const String& keyIdentifier, int windowsVirtualKeyCode, int nativeVirtualKeyCode, int macCharCode, bool handledByInputMethod, bool isAutoRepeat, bool isKeypad, bool isSystemKey);
#elif USE(LIBWPE)
    WebKeyboardEvent(WebEvent&&, const String& text, const String& key, const String& code, const String& keyIdentifier, int windowsVirtualKeyCode, int nativeVirtualKeyCode, bool handledByInputMethod, std::optional<Vector<WebCore::CompositionUnderline>>&&, std::optional<EditingRange>&&, bool isAutoRepeat, bool isKeypad);
#else
    WebKeyboardEvent(WebEvent&&, const String& text, const String& unmodifiedText, const String& key, const String& code, const String& keyIdentifier, int windowsVirtualKeyCode, int nativeVirtualKeyCode, int macCharCode, bool isAutoRepeat, bool isKeypad, bool isSystemKey);
#endif

    const String& text() const { return m_text; }
    const String& unmodifiedText() const { return m_unmodifiedText; }
    const String& key() const { return m_key; }
    const String& code() const { return m_code; }
    const String& keyIdentifier() const { return m_keyIdentifier; }
    int32_t windowsVirtualKeyCode() const { return m_windowsVirtualKeyCode; }
    int32_t nativeVirtualKeyCode() const { return m_nativeVirtualKeyCode; }
    int32_t macCharCode() const { return m_macCharCode; }
#if USE(APPKIT) || PLATFORM(IOS_FAMILY) || PLATFORM(GTK) || USE(LIBWPE)
    bool handledByInputMethod() const { return m_handledByInputMethod; }
#endif
#if PLATFORM(GTK) || USE(LIBWPE)
    const std::optional<Vector<WebCore::CompositionUnderline>>& preeditUnderlines() const { return m_preeditUnderlines; }
    const std::optional<EditingRange>& preeditSelectionRange() const { return m_preeditSelectionRange; }
#endif
#if USE(APPKIT)
    const Vector<WebCore::KeypressCommand>& commands() const { return m_commands; }
#elif PLATFORM(GTK)
    const Vector<String>& commands() const { return m_commands; }
#endif
    bool isAutoRepeat() const { return m_isAutoRepeat; }
    bool isKeypad() const { return m_isKeypad; }
    bool isSystemKey() const { return m_isSystemKey; }

    static bool isKeyboardEventType(WebEventType);

#if PLATFORM(WPE)
    static String keyValueStringForWPEKeyval(unsigned);
    static String keyCodeStringForWPEKeycode(unsigned);
    static String keyIdentifierForWPEKeyval(unsigned);
    static int32_t windowsKeyCodeForWPEKeyval(unsigned);
    static String singleCharacterStringForWPEKeyval(unsigned);
#endif

private:
    String m_text;
    String m_unmodifiedText;
    String m_key;
    String m_code;
    String m_keyIdentifier;
    int32_t m_windowsVirtualKeyCode { 0 };
    int32_t m_nativeVirtualKeyCode { 0 };
    int32_t m_macCharCode { 0 };
#if USE(APPKIT) || PLATFORM(IOS_FAMILY) || PLATFORM(GTK) || USE(LIBWPE)
    bool m_handledByInputMethod { false };
#endif
#if PLATFORM(GTK) || USE(LIBWPE)
    std::optional<Vector<WebCore::CompositionUnderline>> m_preeditUnderlines;
    std::optional<EditingRange> m_preeditSelectionRange;
#endif
#if USE(APPKIT)
    Vector<WebCore::KeypressCommand> m_commands;
#elif PLATFORM(GTK)
    Vector<String> m_commands;
#endif
    bool m_isAutoRepeat { false };
    bool m_isKeypad { false };
    bool m_isSystemKey { false };
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::WebKeyboardEvent)
static bool isType(const WebKit::WebEvent& event) { return WebKit::WebKeyboardEvent::isKeyboardEventType(event.type()); }
SPECIALIZE_TYPE_TRAITS_END()
