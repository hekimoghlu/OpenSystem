/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
#include "PlatformWheelEvent.h"

#include "Scrollbar.h"
#include <wtf/text/TextStream.h>

#if ENABLE(MAC_GESTURE_EVENTS)
#include "PlatformGestureEventMac.h"
#endif

namespace WebCore {

#if ENABLE(MAC_GESTURE_EVENTS)

PlatformWheelEvent PlatformWheelEvent::createFromGesture(const PlatformGestureEvent& platformGestureEvent, double deltaY)
{
    // This tries to match as much of the behavior of `WebKit::WebEventFactory::createWebWheelEvent` as
    // possible assuming `-[NSEvent hasPreciseScrollingDeltas]` and no `-[NSEvent _scrollCount]`.

    double deltaX = 0;
    double wheelTicksX = 0;
    double wheelTicksY = deltaY / static_cast<float>(Scrollbar::pixelsPerLineStep());
    bool shiftKey = platformGestureEvent.modifiers().contains(PlatformEvent::Modifier::ShiftKey);
    bool ctrlKey = true;
    bool altKey = platformGestureEvent.modifiers().contains(PlatformEvent::Modifier::AltKey);
    bool metaKey = platformGestureEvent.modifiers().contains(PlatformEvent::Modifier::MetaKey);
    PlatformWheelEvent platformWheelEvent(platformGestureEvent.pos(), platformGestureEvent.globalPosition(), deltaX, deltaY, wheelTicksX, wheelTicksY, ScrollByPixelWheelEvent, shiftKey, ctrlKey, altKey, metaKey);

    // PlatformEvent
    platformWheelEvent.m_timestamp = platformGestureEvent.timestamp();

    // PlatformWheelEvent
    platformWheelEvent.m_hasPreciseScrollingDeltas = true;

#if ENABLE(KINETIC_SCROLLING)
    switch (platformGestureEvent.type()) {
    case PlatformEvent::Type::GestureStart:
        platformWheelEvent.m_phase = PlatformWheelEventPhase::Began;
        break;
    case PlatformEvent::Type::GestureChange:
        platformWheelEvent.m_phase = PlatformWheelEventPhase::Changed;
        break;
    case PlatformEvent::Type::GestureEnd:
        platformWheelEvent.m_phase = PlatformWheelEventPhase::Ended;
        break;
    default:
        ASSERT_NOT_REACHED();
        break;
    }
#endif // ENABLE(KINETIC_SCROLLING)

#if PLATFORM(COCOA)
    platformWheelEvent.m_ioHIDEventTimestamp = platformWheelEvent.m_timestamp;
    platformWheelEvent.m_rawPlatformDelta = platformWheelEvent.m_rawPlatformDelta;
    platformWheelEvent.m_unacceleratedScrollingDeltaY = deltaY;
#endif // PLATFORM(COCOA)

    return platformWheelEvent;
}

#endif // ENABLE(MAC_GESTURE_EVENTS)

TextStream& operator<<(TextStream& ts, PlatformWheelEventPhase phase)
{
    switch (phase) {
    case PlatformWheelEventPhase::None: ts << "none"; break;
#if ENABLE(KINETIC_SCROLLING)
    case PlatformWheelEventPhase::Began: ts << "began"; break;
    case PlatformWheelEventPhase::Stationary: ts << "stationary"; break;
    case PlatformWheelEventPhase::Changed: ts << "changed"; break;
    case PlatformWheelEventPhase::Ended: ts << "ended"; break;
    case PlatformWheelEventPhase::Cancelled: ts << "cancelled"; break;
    case PlatformWheelEventPhase::MayBegin: ts << "mayBegin"; break;
#endif
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, const PlatformWheelEvent& event)
{
    ts << "PlatformWheelEvent " << &event << " at " << event.position() << " deltaX " << event.deltaX() << " deltaY " << event.deltaY();
    ts << " phase \"" << event.phase() << "\" momentum phase \"" << event.momentumPhase() << "\"";
    ts << " velocity " << event.scrollingVelocity();

    return ts;
}

TextStream& operator<<(TextStream& ts, EventHandling steps)
{
    switch (steps) {
    case EventHandling::DispatchedToDOM: ts << "dispatched to DOM"; break;
    case EventHandling::DefaultPrevented: ts << "default prevented"; break;
    case EventHandling::DefaultHandled: ts << "default handled"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, WheelScrollGestureState state)
{
    switch (state) {
    case WheelScrollGestureState::Blocking: ts << "blocking"; break;
    case WheelScrollGestureState::NonBlocking: ts << "non-blocking"; break;
    }
    return ts;
}

} // namespace WebCore
