/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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

#include "NativeWebWheelEvent.h"
#include <wtf/Deque.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class WebWheelEventCoalescer {
    WTF_MAKE_TZONE_ALLOCATED(WebWheelEventCoalescer);
public:
    // If this returns true, use nextEventToDispatch() to get the event to dispatch.
    bool shouldDispatchEvent(const NativeWebWheelEvent&);
    std::optional<WebWheelEvent> nextEventToDispatch();

    std::optional<NativeWebWheelEvent> takeOldestEventBeingProcessed();

    bool hasEventsBeingProcessed() const { return !m_eventsBeingProcessed.isEmpty(); }
    
    void clear();

private:
    using CoalescedEventSequence = Vector<NativeWebWheelEvent>;

    static bool canCoalesce(const WebWheelEvent&, const WebWheelEvent&);
    static WebWheelEvent coalesce(const WebWheelEvent&, const WebWheelEvent&);

    bool shouldDispatchEventNow(const WebWheelEvent&) const;

    Deque<NativeWebWheelEvent, 2> m_wheelEventQueue;
    Deque<std::unique_ptr<CoalescedEventSequence>> m_eventsBeingProcessed;
};

} // namespace WebKit
