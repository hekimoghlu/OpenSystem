/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#include "WebMouseEvent.h"

#include <WebCore/NavigationAction.h>

namespace WebKit {
using namespace WebCore;

#if PLATFORM(MAC)
WebMouseEvent::WebMouseEvent(WebEvent&& event, WebMouseEventButton button, unsigned short buttons, const IntPoint& positionInView, const IntPoint& globalPosition, float deltaX, float deltaY, float deltaZ, int clickCount, double force, WebMouseEventSyntheticClickType syntheticClickType, int eventNumber, int menuType, GestureWasCancelled gestureWasCancelled, const IntPoint& unadjustedMovementDelta, const Vector<WebMouseEvent>& coalescedEvents, const Vector<WebMouseEvent>& predictedEvents)
#elif PLATFORM(GTK)
WebMouseEvent::WebMouseEvent(WebEvent&& event, WebMouseEventButton button, unsigned short buttons, const IntPoint& positionInView, const IntPoint& globalPosition, float deltaX, float deltaY, float deltaZ, int clickCount, double force, WebMouseEventSyntheticClickType syntheticClickType, PlatformMouseEvent::IsTouch isTouchEvent, WebCore::PointerID pointerId, const String& pointerType, GestureWasCancelled gestureWasCancelled, const IntPoint& unadjustedMovementDelta, const Vector<WebMouseEvent>& coalescedEvents, const Vector<WebMouseEvent>& predictedEvents)
#else
WebMouseEvent::WebMouseEvent(WebEvent&& event, WebMouseEventButton button, unsigned short buttons, const IntPoint& positionInView, const IntPoint& globalPosition, float deltaX, float deltaY, float deltaZ, int clickCount, double force, WebMouseEventSyntheticClickType syntheticClickType, WebCore::PointerID pointerId, const String& pointerType, GestureWasCancelled gestureWasCancelled, const IntPoint& unadjustedMovementDelta, const Vector<WebMouseEvent>& coalescedEvents, const Vector<WebMouseEvent>& predictedEvents)
#endif
    : WebEvent(WTFMove(event))
    , m_button(button)
    , m_buttons(buttons)
    , m_position(positionInView)
    , m_globalPosition(globalPosition)
    , m_deltaX(deltaX)
    , m_deltaY(deltaY)
    , m_deltaZ(deltaZ)
    , m_unadjustedMovementDelta(unadjustedMovementDelta)
    , m_clickCount(clickCount)
#if PLATFORM(MAC)
    , m_eventNumber(eventNumber)
    , m_menuTypeForEvent(menuType)
#elif PLATFORM(GTK)
    , m_isTouchEvent(isTouchEvent)
#endif
    , m_force(force)
    , m_syntheticClickType(syntheticClickType)
#if !PLATFORM(MAC)
    , m_pointerId(pointerId)
    , m_pointerType(pointerType)
#endif
    , m_gestureWasCancelled(gestureWasCancelled)
    , m_coalescedEvents(coalescedEvents)
    , m_predictedEvents(predictedEvents)
{
    ASSERT(isMouseEventType(type()));
}

bool WebMouseEvent::isMouseEventType(WebEventType type)
{
    return type == WebEventType::MouseDown || type == WebEventType::MouseUp || type == WebEventType::MouseMove || type == WebEventType::MouseForceUp || type == WebEventType::MouseForceDown || type == WebEventType::MouseForceChanged;
}

WebMouseEventButton mouseButton(const WebCore::NavigationAction& navigationAction)
{
    auto& mouseEventData = navigationAction.mouseEventData();
    if (mouseEventData && mouseEventData->buttonDown && mouseEventData->isTrusted) {
        switch (mouseEventData->button) {
        case MouseButton::None:
            return WebMouseEventButton::None;

        case MouseButton::Left:
            return WebMouseEventButton::Left;

        case MouseButton::Middle:
            return WebMouseEventButton::Middle;

        case MouseButton::Right:
            return WebMouseEventButton::Right;

        case MouseButton::Other:
        case MouseButton::PointerHasNotChanged: {
            ASSERT_NOT_REACHED();
            return WebMouseEventButton::Left;
        }
        }
    }
    return WebMouseEventButton::None;
}

WebMouseEventSyntheticClickType syntheticClickType(const WebCore::NavigationAction& navigationAction)
{
    auto& mouseEventData = navigationAction.mouseEventData();
    if (mouseEventData && mouseEventData->buttonDown && mouseEventData->isTrusted)
        return static_cast<WebMouseEventSyntheticClickType>(mouseEventData->syntheticClickType);
    return WebMouseEventSyntheticClickType::NoTap;
}

} // namespace WebKit
