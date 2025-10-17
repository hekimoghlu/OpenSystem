/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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
#include "PointerEvent.h"
#include "PlatformMouseEvent.h"

#if ENABLE(TOUCH_EVENTS)

#include "EventNames.h"
#include "PlatformTouchEvent.h"

namespace WebCore {

static const AtomString& pointerEventType(PlatformTouchPoint::State state)
{
    switch (state) {
    case PlatformTouchPoint::State::TouchPressed:
        return eventNames().pointerdownEvent;
    case PlatformTouchPoint::State::TouchMoved:
        return eventNames().pointermoveEvent;
    case PlatformTouchPoint::State::TouchStationary:
        return eventNames().pointermoveEvent;
    case PlatformTouchPoint::State::TouchReleased:
        return eventNames().pointerupEvent;
    case PlatformTouchPoint::State::TouchCancelled:
        return eventNames().pointercancelEvent;
    case PlatformTouchPoint::State::TouchStateEnd:
        break;
    }
    ASSERT_NOT_REACHED();
    return nullAtom();
}

Ref<PointerEvent> PointerEvent::create(const PlatformTouchEvent& event, const Vector<Ref<PointerEvent>>& coalescedEvents, const Vector<Ref<PointerEvent>>& predictedEvents, unsigned index, bool isPrimary, Ref<WindowProxy>&& view, const IntPoint& touchDelta)
{
    const auto& type = pointerEventType(event.touchPoints().at(index).state());
    return adoptRef(*new PointerEvent(type, event, coalescedEvents, predictedEvents, typeCanBubble(type), typeIsCancelable(type), index, isPrimary, WTFMove(view), touchDelta));
}

Ref<PointerEvent> PointerEvent::create(const PlatformTouchEvent& event, const Vector<Ref<PointerEvent>>& coalescedEvents, const Vector<Ref<PointerEvent>>& predictedEvents, CanBubble canBubble, IsCancelable isCancelable, unsigned index, bool isPrimary, Ref<WindowProxy>&& view, const IntPoint& touchDelta)
{
    const auto& type = pointerEventType(event.touchPoints().at(index).state());
    return adoptRef(*new PointerEvent(type, event, coalescedEvents, predictedEvents, canBubble, isCancelable, index, isPrimary, WTFMove(view), touchDelta));
}

Ref<PointerEvent> PointerEvent::create(const AtomString& type, const PlatformTouchEvent& event, const Vector<Ref<PointerEvent>>& coalescedEvents, const Vector<Ref<PointerEvent>>& predictedEvents, unsigned index, bool isPrimary, Ref<WindowProxy>&& view, const IntPoint& touchDelta)
{
    return adoptRef(*new PointerEvent(type, event, coalescedEvents, predictedEvents, typeCanBubble(type), typeIsCancelable(type), index, isPrimary, WTFMove(view), touchDelta));
}

// According to the PointerEvents spec all active pointer ids have to be unique.
// Libinput on Linux assigns the ids of the touchpoints starting at 0, but
// the ids 0 and 1 are used for the pointer ids of mouse and pen/stylus.
const unsigned touchMinimumPointerId = WebCore::mousePointerID + 1;

PointerEvent::PointerEvent(const AtomString& type, const PlatformTouchEvent& event, const Vector<Ref<PointerEvent>>& coalescedEvents, const Vector<Ref<PointerEvent>>& predictedEvents, CanBubble canBubble, IsCancelable isCancelable, unsigned index, bool isPrimary, Ref<WindowProxy>&& view, const IntPoint& touchDelta)
    : MouseEvent(EventInterfaceType::PointerEvent, type, canBubble, isCancelable, typeIsComposed(type), event.timestamp().approximateMonotonicTime(), WTFMove(view), 0, event.touchPoints().at(index).pos(), event.touchPoints().at(index).pos(), touchDelta.x(), touchDelta.y(), event.modifiers(), buttonForType(type), buttonsForType(type), nullptr, 0, SyntheticClickType::NoTap, { }, { }, IsSimulated::No, IsTrusted::Yes)
    , m_pointerId(touchMinimumPointerId + event.touchPoints().at(index).id())
    , m_width(2 * event.touchPoints().at(index).radiusX())
    , m_height(2 * event.touchPoints().at(index).radiusY())
    , m_pressure(event.touchPoints().at(index).force())
    , m_pointerType(touchPointerEventType())
    , m_isPrimary(isPrimary)
    , m_coalescedEvents(coalescedEvents)
    , m_predictedEvents(predictedEvents)
{
}

} // namespace WebCore

#endif // ENABLE(TOUCH_EVENTS)
