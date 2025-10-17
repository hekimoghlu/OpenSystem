/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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

#include "IntPoint.h"
#include "PlatformEvent.h"
#include "PointerEventTypeNames.h"
#include "PointerID.h"
#include <wtf/UUID.h>
#include <wtf/WindowsExtras.h>

namespace WebCore {

const double ForceAtClick = 1;
const double ForceAtForceClick = 2;

// These button numbers match the ones used in the DOM API, 0 through 2, except for None and Other which aren't specified.
// We reserve -2 for the former and -1 to represent pointer events that indicate that the pressed mouse button hasn't
// changed since the last event, as specified in the DOM API for Pointer Events.
// https://w3c.github.io/uievents/#dom-mouseevent-button
// https://w3c.github.io/pointerevents/#the-button-property
enum class MouseButton : int8_t { None = -2, PointerHasNotChanged, Left, Middle, Right, Other };
enum class SyntheticClickType : uint8_t { NoTap, OneFingerTap, TwoFingerTap };

class PlatformMouseEvent : public PlatformEvent {
public:
    PlatformMouseEvent()
        : PlatformEvent(Type::MouseMoved)
    {
    }

    PlatformMouseEvent(const IntPoint& position, const IntPoint& globalPosition, MouseButton button, PlatformEvent::Type type, int clickCount, OptionSet<PlatformEvent::Modifier> modifiers, WallTime timestamp, double force, SyntheticClickType syntheticClickType, PointerID pointerId = mousePointerID)
        : PlatformEvent(type, modifiers, timestamp)
        , m_button(button)
        , m_syntheticClickType(syntheticClickType)
        , m_position(position)
        , m_globalPosition(globalPosition)
        , m_force(force)
        , m_pointerId(pointerId)
        , m_clickCount(clickCount)
    {
    }

    // This position is relative to the enclosing NSWindow in WebKit1, and is WKWebView-relative in WebKit2.
    // Use ScrollView::windowToContents() to convert it to into the contents of a given view.
    const IntPoint& position() const { return m_position; }
    const IntPoint& globalPosition() const { return m_globalPosition; }
    const IntPoint& movementDelta() const { return m_movementDelta; }
    // Unaccelerated pointer movement
    const IntPoint& unadjustedMovementDelta() const { return m_unadjustedMovementDelta; }

    MouseButton button() const { return m_button; }
    unsigned short buttons() const { return m_buttons; }
    int clickCount() const { return m_clickCount; }
    unsigned modifierFlags() const { return m_modifierFlags; }
    double force() const { return m_force; }
    SyntheticClickType syntheticClickType() const { return m_syntheticClickType; }
    PointerID pointerId() const { return m_pointerId; }
    const String& pointerType() const { return m_pointerType; }

    Vector<PlatformMouseEvent> coalescedEvents() const { return m_coalescedEvents; }
    Vector<PlatformMouseEvent> predictedEvents() const { return m_predictedEvents; }

#if PLATFORM(MAC)
    int eventNumber() const { return m_eventNumber; }
    int menuTypeForEvent() const { return m_menuTypeForEvent; }
#endif

#if PLATFORM(WIN)
    WEBCORE_EXPORT PlatformMouseEvent(HWND, UINT, WPARAM, LPARAM, bool didActivateWebView = false);
    void setClickCount(int count) { m_clickCount = count; }
    bool didActivateWebView() const { return m_didActivateWebView; }
#endif

#if PLATFORM(GTK)
    enum class IsTouch : bool { No, Yes };

    bool isTouchEvent() const { return m_isTouchEvent == IsTouch::Yes; }
#endif

protected:
    MouseButton m_button { MouseButton::None };
    SyntheticClickType m_syntheticClickType { SyntheticClickType::NoTap };

    IntPoint m_position;
    IntPoint m_globalPosition;
    IntPoint m_movementDelta;
    IntPoint m_unadjustedMovementDelta;
    double m_force { 0 };
    PointerID m_pointerId { mousePointerID };
    String m_pointerType { mousePointerEventType() };
    int m_clickCount { 0 };
    unsigned m_modifierFlags { 0 };
    unsigned short m_buttons { 0 };
    Vector<PlatformMouseEvent> m_coalescedEvents;
    Vector<PlatformMouseEvent> m_predictedEvents;
#if PLATFORM(MAC)
    int m_eventNumber { 0 };
    int m_menuTypeForEvent { 0 };
#elif PLATFORM(WIN)
    bool m_didActivateWebView { false };
#elif PLATFORM(GTK)
    IsTouch m_isTouchEvent { IsTouch::No };
#endif
};

} // namespace WebCore
