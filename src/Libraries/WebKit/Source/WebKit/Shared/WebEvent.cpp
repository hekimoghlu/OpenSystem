/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 14, 2021.
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
#include "WebEvent.h"

#include "Decoder.h"
#include "Encoder.h"
#include "WebKeyboardEvent.h"
#include <WebCore/WindowsKeyboardCodes.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebEvent);

WebEvent::WebEvent(WebEventType type, OptionSet<WebEventModifier> modifiers, WallTime timestamp, WTF::UUID authorizationToken)
    : m_type(type)
    , m_modifiers(modifiers)
    , m_timestamp(timestamp)
    , m_authorizationToken(authorizationToken)
{
}

WebEvent::WebEvent(WebEventType type, OptionSet<WebEventModifier> modifiers, WallTime timestamp)
    : m_type(type)
    , m_modifiers(modifiers)
    , m_timestamp(timestamp)
    , m_authorizationToken(WTF::UUID::createVersion4())
{
}

// https://html.spec.whatwg.org/multipage/interaction.html#activation-triggering-input-event
bool WebEvent::isActivationTriggeringEvent() const
{
    switch (type()) {
    case WebEventType::MouseDown:
#if ENABLE(TOUCH_EVENTS)
    case WebEventType::TouchEnd:
#endif
        return true;
    case WebEventType::KeyDown:
        return downcast<WebKeyboardEvent>(*this).windowsVirtualKeyCode() != VK_ESCAPE;
    default:
        break;
    }
    return false;
}

TextStream& operator<<(TextStream& ts, WebEventType eventType)
{
    switch (eventType) {
    case WebEventType::MouseDown: ts << "MouseDown"; break;
    case WebEventType::MouseUp: ts << "MouseUp"; break;
    case WebEventType::MouseMove: ts << "MouseMove"; break;
    case WebEventType::MouseForceChanged: ts << "MouseForceChanged"; break;
    case WebEventType::MouseForceDown: ts << "MouseForceDown"; break;
    case WebEventType::MouseForceUp: ts << "MouseForceUp"; break;
    case WebEventType::Wheel: ts << "Wheel"; break;
    case WebEventType::KeyDown: ts << "KeyDown"; break;
    case WebEventType::KeyUp: ts << "KeyUp"; break;
    case WebEventType::RawKeyDown: ts << "RawKeyDown"; break;
    case WebEventType::Char: ts << "Char"; break;

#if ENABLE(TOUCH_EVENTS)
    case WebEventType::TouchStart: ts << "TouchStart"; break;
    case WebEventType::TouchMove: ts << "TouchMove"; break;
    case WebEventType::TouchEnd: ts << "TouchEnd"; break;
    case WebEventType::TouchCancel: ts << "TouchCancel"; break;
#endif

#if ENABLE(MAC_GESTURE_EVENTS)
    case WebEventType::GestureStart: ts << "GestureStart"; break;
    case WebEventType::GestureChange: ts << "GestureChange"; break;
    case WebEventType::GestureEnd: ts << "GestureEnd"; break;
#endif
    }

    return ts;
}

} // namespace WebKit
