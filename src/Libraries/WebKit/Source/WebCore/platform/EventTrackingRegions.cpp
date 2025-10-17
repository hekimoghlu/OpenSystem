/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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
#include "EventTrackingRegions.h"

#include "EventNames.h"

namespace WebCore {

ASCIILiteral EventTrackingRegions::eventName(EventType eventType)
{
    switch (eventType) {
    case EventType::Mousedown:
        return "mousedown"_s;
    case EventType::Mousemove:
        return "mousemove"_s;
    case EventType::Mouseup:
        return "mouseup"_s;
    case EventType::Mousewheel:
        return "mousewheel"_s;
    case EventType::Pointerdown:
        return "pointerdown"_s;
    case EventType::Pointerenter:
        return "pointerenter"_s;
    case EventType::Pointerleave:
        return "pointerleave"_s;
    case EventType::Pointermove:
        return "pointermove"_s;
    case EventType::Pointerout:
        return "pointerout"_s;
    case EventType::Pointerover:
        return "pointerover"_s;
    case EventType::Pointerup:
        return "pointerup"_s;
    case EventType::Touchend:
        return "touchend"_s;
    case EventType::Touchforcechange:
        return "touchforcechange"_s;
    case EventType::Touchmove:
        return "touchmove"_s;
    case EventType::Touchstart:
        return "touchstart"_s;
    case EventType::Wheel:
        return "wheel"_s;
    }
    return ASCIILiteral();
}

const AtomString& EventTrackingRegions::eventNameAtomString(const EventNames& eventNames, EventType eventType)
{
    switch (eventType) {
    case EventType::Mousedown:
        return eventNames.mousedownEvent;
    case EventType::Mousemove:
        return eventNames.mousemoveEvent;
    case EventType::Mouseup:
        return eventNames.mouseupEvent;
    case EventType::Mousewheel:
        return eventNames.mousewheelEvent;
    case EventType::Pointerdown:
        return eventNames.pointerdownEvent;
    case EventType::Pointerenter:
        return eventNames.pointerenterEvent;
    case EventType::Pointerleave:
        return eventNames.pointerleaveEvent;
    case EventType::Pointermove:
        return eventNames.pointermoveEvent;
    case EventType::Pointerout:
        return eventNames.pointeroutEvent;
    case EventType::Pointerover:
        return eventNames.pointeroverEvent;
    case EventType::Pointerup:
        return eventNames.pointerupEvent;
    case EventType::Touchend:
        return eventNames.touchendEvent;
    case EventType::Touchforcechange:
        return eventNames.touchforcechangeEvent;
    case EventType::Touchmove:
        return eventNames.touchmoveEvent;
    case EventType::Touchstart:
        return eventNames.touchstartEvent;
    case EventType::Wheel:
        return eventNames.wheelEvent;
    }
    return nullAtom();
}

TrackingType EventTrackingRegions::trackingTypeForPoint(EventType eventType, const IntPoint& point)
{
    auto synchronousRegionIterator = eventSpecificSynchronousDispatchRegions.find(eventType);
    if (synchronousRegionIterator != eventSpecificSynchronousDispatchRegions.end()) {
        if (synchronousRegionIterator->value.contains(point))
            return TrackingType::Synchronous;
    }

    if (asynchronousDispatchRegion.contains(point))
        return TrackingType::Asynchronous;
    return TrackingType::NotTracking;
}

bool EventTrackingRegions::isEmpty() const
{
    return asynchronousDispatchRegion.isEmpty() && eventSpecificSynchronousDispatchRegions.isEmpty();
}

void EventTrackingRegions::translate(IntSize offset)
{
    asynchronousDispatchRegion.translate(offset);
    for (auto& slot : eventSpecificSynchronousDispatchRegions)
        slot.value.translate(offset);
}

void EventTrackingRegions::uniteSynchronousRegion(EventType eventType, const Region& region)
{
    if (region.isEmpty())
        return;

    auto addResult = eventSpecificSynchronousDispatchRegions.add(eventType, region);
    if (!addResult.isNewEntry)
        addResult.iterator->value.unite(region);
}

void EventTrackingRegions::unite(const EventTrackingRegions& eventTrackingRegions)
{
    asynchronousDispatchRegion.unite(eventTrackingRegions.asynchronousDispatchRegion);
    for (auto& slot : eventTrackingRegions.eventSpecificSynchronousDispatchRegions)
        uniteSynchronousRegion(slot.key, slot.value);
}

} // namespace WebCore
