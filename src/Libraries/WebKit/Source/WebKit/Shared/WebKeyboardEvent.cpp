/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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
#include "WebKeyboardEvent.h"

#include <WebCore/KeypressCommand.h>

namespace WebKit {

#if USE(APPKIT)

WebKeyboardEvent::WebKeyboardEvent(WebEvent&& event, const String& text, const String& unmodifiedText, const String& key, const String& code, const String& keyIdentifier, int windowsVirtualKeyCode, int nativeVirtualKeyCode, int macCharCode, bool handledByInputMethod, const Vector<WebCore::KeypressCommand>& commands, bool isAutoRepeat, bool isKeypad, bool isSystemKey)
    : WebEvent(WTFMove(event))
    , m_text(text)
    , m_unmodifiedText(unmodifiedText)
    , m_key(key)
    , m_code(code)
    , m_keyIdentifier(keyIdentifier)
    , m_windowsVirtualKeyCode(windowsVirtualKeyCode)
    , m_nativeVirtualKeyCode(nativeVirtualKeyCode)
    , m_macCharCode(macCharCode)
    , m_handledByInputMethod(handledByInputMethod)
    , m_commands(commands)
    , m_isAutoRepeat(isAutoRepeat)
    , m_isKeypad(isKeypad)
    , m_isSystemKey(isSystemKey)
{
    ASSERT(isKeyboardEventType(type()));
}

#elif PLATFORM(GTK)

WebKeyboardEvent::WebKeyboardEvent(WebEvent&& event, const String& text, const String& key, const String& code, const String& keyIdentifier, int windowsVirtualKeyCode, int nativeVirtualKeyCode, bool handledByInputMethod, std::optional<Vector<WebCore::CompositionUnderline>>&& preeditUnderlines, std::optional<EditingRange>&& preeditSelectionRange, Vector<String>&& commands, bool isAutoRepeat, bool isKeypad)
    : WebEvent(WTFMove(event))
    , m_text(text)
    , m_unmodifiedText(text)
    , m_key(key)
    , m_code(code)
    , m_keyIdentifier(keyIdentifier)
    , m_windowsVirtualKeyCode(windowsVirtualKeyCode)
    , m_nativeVirtualKeyCode(nativeVirtualKeyCode)
    , m_macCharCode(0)
    , m_handledByInputMethod(handledByInputMethod)
    , m_preeditUnderlines(WTFMove(preeditUnderlines))
    , m_preeditSelectionRange(WTFMove(preeditSelectionRange))
    , m_commands(WTFMove(commands))
    , m_isAutoRepeat(isAutoRepeat)
    , m_isKeypad(isKeypad)
    , m_isSystemKey(false)
{
    ASSERT(isKeyboardEventType(type()));
}

#elif PLATFORM(IOS_FAMILY)

WebKeyboardEvent::WebKeyboardEvent(WebEvent&& event, const String& text, const String& unmodifiedText, const String& key, const String& code, const String& keyIdentifier, int windowsVirtualKeyCode, int nativeVirtualKeyCode, int macCharCode, bool handledByInputMethod, bool isAutoRepeat, bool isKeypad, bool isSystemKey)
    : WebEvent(WTFMove(event))
    , m_text(text)
    , m_unmodifiedText(unmodifiedText)
    , m_key(key)
    , m_code(code)
    , m_keyIdentifier(keyIdentifier)
    , m_windowsVirtualKeyCode(windowsVirtualKeyCode)
    , m_nativeVirtualKeyCode(nativeVirtualKeyCode)
    , m_macCharCode(macCharCode)
    , m_handledByInputMethod(handledByInputMethod)
    , m_isAutoRepeat(isAutoRepeat)
    , m_isKeypad(isKeypad)
    , m_isSystemKey(isSystemKey)
{
    ASSERT(isKeyboardEventType(type()));
}

#elif USE(LIBWPE)

WebKeyboardEvent::WebKeyboardEvent(WebEvent&& event, const String& text, const String& key, const String& code, const String& keyIdentifier, int windowsVirtualKeyCode, int nativeVirtualKeyCode, bool handledByInputMethod, std::optional<Vector<WebCore::CompositionUnderline>>&& preeditUnderlines, std::optional<EditingRange>&& preeditSelectionRange, bool isAutoRepeat, bool isKeypad)
    : WebEvent(WTFMove(event))
    , m_text(text)
    , m_unmodifiedText(text)
    , m_key(key)
    , m_code(code)
    , m_keyIdentifier(keyIdentifier)
    , m_windowsVirtualKeyCode(windowsVirtualKeyCode)
    , m_nativeVirtualKeyCode(nativeVirtualKeyCode)
    , m_macCharCode(0)
    , m_handledByInputMethod(handledByInputMethod)
    , m_preeditUnderlines(WTFMove(preeditUnderlines))
    , m_preeditSelectionRange(WTFMove(preeditSelectionRange))
    , m_isAutoRepeat(isAutoRepeat)
    , m_isKeypad(isKeypad)
    , m_isSystemKey(false)
{
    ASSERT(isKeyboardEventType(type()));
}

#else

WebKeyboardEvent::WebKeyboardEvent(WebEvent&& event, const String& text, const String& unmodifiedText, const String& key, const String& code, const String& keyIdentifier, int windowsVirtualKeyCode, int nativeVirtualKeyCode, int macCharCode, bool isAutoRepeat, bool isKeypad, bool isSystemKey)
    : WebEvent(WTFMove(event))
    , m_text(text)
    , m_unmodifiedText(unmodifiedText)
    , m_key(key)
    , m_code(code)
    , m_keyIdentifier(keyIdentifier)
    , m_windowsVirtualKeyCode(windowsVirtualKeyCode)
    , m_nativeVirtualKeyCode(nativeVirtualKeyCode)
    , m_macCharCode(macCharCode)
    , m_isAutoRepeat(isAutoRepeat)
    , m_isKeypad(isKeypad)
    , m_isSystemKey(isSystemKey)
{
    ASSERT(isKeyboardEventType(type()));
}

#endif

WebKeyboardEvent::~WebKeyboardEvent()
{
}

bool WebKeyboardEvent::isKeyboardEventType(WebEventType type)
{
    return type == WebEventType::RawKeyDown || type == WebEventType::KeyDown || type == WebEventType::KeyUp || type == WebEventType::Char;
}

} // namespace WebKit
