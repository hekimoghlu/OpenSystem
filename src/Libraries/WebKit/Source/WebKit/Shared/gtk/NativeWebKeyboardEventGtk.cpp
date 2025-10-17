/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#include "NativeWebKeyboardEvent.h"

#include "WebEventFactory.h"
#include <WebCore/GtkVersioning.h>

namespace WebKit {

#if USE(GTK4)
#define constructNativeEvent(event) event
#else
#define constructNativeEvent(event) gdk_event_copy(event)
#endif

NativeWebKeyboardEvent::NativeWebKeyboardEvent(GdkEvent* event, const String& text, bool isAutoRepeat, Vector<String>&& commands)
    : WebKeyboardEvent(WebEventFactory::createWebKeyboardEvent(event, text, isAutoRepeat, false, std::nullopt, std::nullopt, WTFMove(commands)))
    , m_nativeEvent(constructNativeEvent(event))
{
}

NativeWebKeyboardEvent::NativeWebKeyboardEvent(const String& text, std::optional<Vector<WebCore::CompositionUnderline>>&& preeditUnderlines, std::optional<EditingRange>&& preeditSelectionRange)
    : WebKeyboardEvent(WebEvent(WebEventType::KeyDown, { }, WallTime::now()), text, "Unidentified"_s, "Unidentified"_s, "U+0000"_s, 229, GDK_KEY_VoidSymbol, true, WTFMove(preeditUnderlines), WTFMove(preeditSelectionRange), { }, false, false)
{
}

NativeWebKeyboardEvent::NativeWebKeyboardEvent(WebEventType type, const String& text, const String& key, const String& code, const String& keyIdentifier, int windowsVirtualKeyCode, int nativeVirtualKeyCode, Vector<String>&& commands, bool isAutoRepeat, bool isKeypad, OptionSet<WebEventModifier> modifiers)
    : WebKeyboardEvent(WebEvent(type, modifiers, WallTime::now()), text, key, code, keyIdentifier, windowsVirtualKeyCode, nativeVirtualKeyCode, false, std::nullopt, std::nullopt, WTFMove(commands), isAutoRepeat, isKeypad)
{
}

NativeWebKeyboardEvent::NativeWebKeyboardEvent(const NativeWebKeyboardEvent& event)
    : WebKeyboardEvent(WebEvent(event.type(), event.modifiers(), event.timestamp()), event.text(), event.key(), event.code(), event.keyIdentifier(), event.windowsVirtualKeyCode(), event.nativeVirtualKeyCode(), event.handledByInputMethod(), std::optional<Vector<WebCore::CompositionUnderline>>(event.preeditUnderlines()), std::optional<EditingRange>(event.preeditSelectionRange()), Vector<String>(event.commands()), event.isAutoRepeat(), event.isKeypad())
    , m_nativeEvent(event.nativeEvent() ? constructNativeEvent(event.nativeEvent()) : nullptr)
{
}

} // namespace WebKit

#undef constructNativeEvent
