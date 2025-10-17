/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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

#include "Region.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct EventNames;

enum class TrackingType : uint8_t {
    NotTracking = 0,
    Asynchronous = 1,
    Synchronous = 2
};

enum class EventTrackingRegionsEventType : uint8_t {
    Mousedown,
    Mousemove,
    Mouseup,
    Mousewheel,
    Pointerdown,
    Pointerenter,
    Pointerleave,
    Pointermove,
    Pointerout,
    Pointerover,
    Pointerup,
    Touchend,
    Touchforcechange,
    Touchmove,
    Touchstart,
    Wheel,
};

struct EventTrackingRegions {
    using EventType = EventTrackingRegionsEventType;

    WEBCORE_EXPORT static ASCIILiteral eventName(EventType);
    WEBCORE_EXPORT static const AtomString& eventNameAtomString(const EventNames&, EventType);

    // Region for which events can be dispatched without blocking scrolling.
    Region asynchronousDispatchRegion;

    // Regions for which events must be sent before performing the default behavior.
    // The key is the EventType with an active handler.
    using EventSpecificSynchronousDispatchRegions = HashMap<EventType, Region, WTF::IntHash<EventType>, WTF::StrongEnumHashTraits<EventType>>;
    EventSpecificSynchronousDispatchRegions eventSpecificSynchronousDispatchRegions;

    bool isEmpty() const;

    void translate(IntSize);
    void uniteSynchronousRegion(EventType, const Region&);
    void unite(const EventTrackingRegions&);

    TrackingType trackingTypeForPoint(EventType, const IntPoint&);

    friend bool operator==(const EventTrackingRegions&, const EventTrackingRegions&) = default;
};

} // namespace WebCore
