/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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

// FIXME: We should probably move to making the WebCore/PlatformFooEvents trivial classes so that
// we can use them as the event type.

#include "WebEvent.h"
#include <WebCore/IntPoint.h>
#include <WebCore/PlatformMouseEvent.h>
#include <WebCore/PointerEventTypeNames.h>
#include <WebCore/PointerID.h>

namespace WebCore {
class NavigationAction;
}

namespace WebKit {

enum class GestureWasCancelled : bool { No, Yes };

enum class WebMouseEventButton : int8_t {
    Left,
    Middle,
    Right,
    None = -2,
};
WebMouseEventButton mouseButton(const WebCore::NavigationAction&);

enum class WebMouseEventSyntheticClickType : uint8_t {
    NoTap,
    OneFingerTap,
    TwoFingerTap
};
WebMouseEventSyntheticClickType syntheticClickType(const WebCore::NavigationAction&);

class WebMouseEvent : public WebEvent {
public:
#if PLATFORM(MAC)
    WebMouseEvent(WebEvent&&, WebMouseEventButton, unsigned short buttons, const WebCore::IntPoint& positionInView, const WebCore::IntPoint& globalPosition, float deltaX, float deltaY, float deltaZ, int clickCount, double force, WebMouseEventSyntheticClickType = WebMouseEventSyntheticClickType::NoTap, int eventNumber = -1, int menuType = 0, GestureWasCancelled = GestureWasCancelled::No, const WebCore::IntPoint& unadjustedMovementDelta = { }, const Vector<WebMouseEvent>& coalescedEvents = { }, const Vector<WebMouseEvent>& predictedEvents = { });
#elif PLATFORM(GTK)
    WebMouseEvent(WebEvent&&, WebMouseEventButton, unsigned short buttons, const WebCore::IntPoint& positionInView, const WebCore::IntPoint& globalPosition, float deltaX, float deltaY, float deltaZ, int clickCount, double force = 0, WebMouseEventSyntheticClickType = WebMouseEventSyntheticClickType::NoTap, WebCore::PlatformMouseEvent::IsTouch m_isTouchEvent = WebCore::PlatformMouseEvent::IsTouch::No, WebCore::PointerID = WebCore::mousePointerID, const String& pointerType = WebCore::mousePointerEventType(), GestureWasCancelled = GestureWasCancelled::No, const WebCore::IntPoint& unadjustedMovementDelta = { }, const Vector<WebMouseEvent>& coalescedEvents = { }, const Vector<WebMouseEvent>& predictedEvents = { });
#else
    WebMouseEvent(WebEvent&&, WebMouseEventButton, unsigned short buttons, const WebCore::IntPoint& positionInView, const WebCore::IntPoint& globalPosition, float deltaX, float deltaY, float deltaZ, int clickCount, double force = 0, WebMouseEventSyntheticClickType = WebMouseEventSyntheticClickType::NoTap, WebCore::PointerID = WebCore::mousePointerID, const String& pointerType = WebCore::mousePointerEventType(), GestureWasCancelled = GestureWasCancelled::No, const WebCore::IntPoint& unadjustedMovementDelta = { }, const Vector<WebMouseEvent>& coalescedEvents = { }, const Vector<WebMouseEvent>& predictedEvents = { });
#endif

    WebMouseEventButton button() const { return m_button; }
    unsigned short buttons() const { return m_buttons; }
    const WebCore::IntPoint& position() const { return m_position; } // Relative to the view.
    void setPosition(const WebCore::IntPoint& position) { m_position = position; }
    const WebCore::IntPoint& globalPosition() const { return m_globalPosition; }
    float deltaX() const { return m_deltaX; }
    float deltaY() const { return m_deltaY; }
    float deltaZ() const { return m_deltaZ; }
    int32_t clickCount() const { return m_clickCount; }
#if PLATFORM(MAC)
    int32_t eventNumber() const { return m_eventNumber; }
    int32_t menuTypeForEvent() const { return m_menuTypeForEvent; }
#elif PLATFORM(GTK)
    WebCore::PlatformMouseEvent::IsTouch isTouchEvent() const { return m_isTouchEvent; }
#endif
    double force() const { return m_force; }
    WebMouseEventSyntheticClickType syntheticClickType() const { return m_syntheticClickType; }
    WebCore::PointerID pointerId() const { return m_pointerId; }
    const String& pointerType() const { return m_pointerType; }
    GestureWasCancelled gestureWasCancelled() const { return m_gestureWasCancelled; }
    // Unaccelerated pointer movement
    const WebCore::IntPoint& unadjustedMovementDelta() const { return m_unadjustedMovementDelta; }

    void setCoalescedEvents(const Vector<WebMouseEvent>& coalescedEvents) { m_coalescedEvents = coalescedEvents; }
    Vector<WebMouseEvent> coalescedEvents() const { return m_coalescedEvents; }

    void setPredictedEvents(const Vector<WebMouseEvent>& predictedEvents) { m_predictedEvents = predictedEvents; }
    Vector<WebMouseEvent> predictedEvents() const { return m_predictedEvents; }

private:
    static bool isMouseEventType(WebEventType);

    WebMouseEventButton m_button { WebMouseEventButton::None };
    unsigned short m_buttons { 0 };
    WebCore::IntPoint m_position; // Relative to the view.
    WebCore::IntPoint m_globalPosition;
    float m_deltaX { 0 };
    float m_deltaY { 0 };
    float m_deltaZ { 0 };
    WebCore::IntPoint m_unadjustedMovementDelta;
    int32_t m_clickCount { 0 };
#if PLATFORM(MAC)
    int32_t m_eventNumber { -1 };
    int32_t m_menuTypeForEvent { 0 };
#elif PLATFORM(GTK)
    WebCore::PlatformMouseEvent::IsTouch m_isTouchEvent { WebCore::PlatformMouseEvent::IsTouch::No };
#endif
    double m_force { 0 };
    WebMouseEventSyntheticClickType m_syntheticClickType { WebMouseEventSyntheticClickType::NoTap };
    WebCore::PointerID m_pointerId { WebCore::mousePointerID };
    String m_pointerType { WebCore::mousePointerEventType() };
    GestureWasCancelled m_gestureWasCancelled { GestureWasCancelled::No };
    Vector<WebMouseEvent> m_coalescedEvents;
    Vector<WebMouseEvent> m_predictedEvents;
};

} // namespace WebKit
