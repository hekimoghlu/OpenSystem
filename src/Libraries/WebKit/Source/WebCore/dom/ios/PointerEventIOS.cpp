/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 10, 2023.
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
#import "config.h"
#import "PointerEvent.h"

#if ENABLE(TOUCH_EVENTS) && PLATFORM(IOS_FAMILY)

#import "EventNames.h"
#import "PlatformMouseEvent.h"

namespace WebCore {

static const AtomString& pointerEventType(PlatformTouchPoint::TouchPhaseType phase)
{
    switch (phase) {
    case PlatformTouchPoint::TouchPhaseBegan:
        return eventNames().pointerdownEvent;
    case PlatformTouchPoint::TouchPhaseMoved:
        return eventNames().pointermoveEvent;
    case PlatformTouchPoint::TouchPhaseStationary:
        return eventNames().pointermoveEvent;
    case PlatformTouchPoint::TouchPhaseEnded:
        return eventNames().pointerupEvent;
    case PlatformTouchPoint::TouchPhaseCancelled:
        return eventNames().pointercancelEvent;
    }
    ASSERT_NOT_REACHED();
    return nullAtom();
}

Ref<PointerEvent> PointerEvent::create(const PlatformTouchEvent& event, const Vector<Ref<PointerEvent>>& coalescedEvents, const Vector<Ref<PointerEvent>>& predictedEvents, unsigned index, bool isPrimary, Ref<WindowProxy>&& view, const IntPoint& touchDelta)
{
    const auto& type = pointerEventType(event.touchPhaseAtIndex(index));

    return adoptRef(*new PointerEvent(type, event, coalescedEvents, predictedEvents, typeCanBubble(type), typeIsCancelable(type), index, isPrimary, WTFMove(view), touchDelta));
}

Ref<PointerEvent> PointerEvent::create(const PlatformTouchEvent& event, const Vector<Ref<PointerEvent>>& coalescedEvents, const Vector<Ref<PointerEvent>>& predictedEvents, CanBubble canBubble, IsCancelable isCancelable, unsigned index, bool isPrimary, Ref<WindowProxy>&& view, const IntPoint& touchDelta)
{
    const auto& type = pointerEventType(event.touchPhaseAtIndex(index));

    return adoptRef(*new PointerEvent(type, event, coalescedEvents, predictedEvents, canBubble, isCancelable, index, isPrimary, WTFMove(view), touchDelta));
}

Ref<PointerEvent> PointerEvent::create(const AtomString& type, const PlatformTouchEvent& event, const Vector<Ref<PointerEvent>>& coalescedEvents, const Vector<Ref<PointerEvent>>& predictedEvents, unsigned index, bool isPrimary, Ref<WindowProxy>&& view, const IntPoint& touchDelta)
{
    return adoptRef(*new PointerEvent(type, event, coalescedEvents, predictedEvents, typeCanBubble(type), typeIsCancelable(type), index, isPrimary, WTFMove(view), touchDelta));
}

PointerEvent::PointerEvent(const AtomString& type, const PlatformTouchEvent& event, const Vector<Ref<PointerEvent>>& coalescedEvents, const Vector<Ref<PointerEvent>>& predictedEvents, CanBubble canBubble, IsCancelable isCancelable, unsigned index, bool isPrimary, Ref<WindowProxy>&& view, const IntPoint& touchDelta)
    : MouseEvent(EventInterfaceType::PointerEvent, type, canBubble, isCancelable, typeIsComposed(type), event.timestamp().approximateMonotonicTime(), WTFMove(view), 0,
        event.touchLocationInRootViewAtIndex(index), event.touchLocationInRootViewAtIndex(index), touchDelta.x(), touchDelta.y(), event.modifiers(), buttonForType(type), buttonsForType(type), nullptr, 0, SyntheticClickType::NoTap, { }, { }, IsSimulated::No, IsTrusted::Yes)
    , m_pointerId(event.touchIdentifierAtIndex(index))
    , m_width(2 * event.radiusXAtIndex(index))
    , m_height(2 * event.radiusYAtIndex(index))
    , m_pressure(event.forceAtIndex(index))
    , m_pointerType(event.touchTypeAtIndex(index) == PlatformTouchPoint::TouchType::Stylus ? penPointerEventType() : touchPointerEventType())
    , m_isPrimary(isPrimary)
    , m_coalescedEvents(coalescedEvents)
    , m_predictedEvents(predictedEvents)
{
    m_azimuthAngle = event.azimuthAngleAtIndex(index);
    m_altitudeAngle = m_pointerType == penPointerEventType() ? event.altitudeAngleAtIndex(index) : piOverTwoDouble;
    auto tilt = tiltFromAngle(m_altitudeAngle, m_azimuthAngle);
    m_tiltX = tilt.tiltX;
    m_tiltY = tilt.tiltY;
}

} // namespace WebCore

#endif // ENABLE(TOUCH_EVENTS) && PLATFORM(IOS_FAMILY)
