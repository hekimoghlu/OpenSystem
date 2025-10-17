/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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

#include "GCReachableRef.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class Event;
class EventQueueScope;

class ScopedEventQueue {
    WTF_MAKE_TZONE_ALLOCATED(ScopedEventQueue);
    WTF_MAKE_NONCOPYABLE(ScopedEventQueue);
public:
    static ScopedEventQueue& singleton();

    struct ScopedEvent {
        Ref<Event> event;
        GCReachableRef<Node> target;
    };
    void enqueueEvent(ScopedEvent&&);

private:
    ScopedEventQueue() = default;
    ~ScopedEventQueue() = delete;

    void dispatchEvent(const ScopedEvent&) const;
    void dispatchAllEvents();
    void incrementScopingLevel() { ++m_scopingLevel; }
    void decrementScopingLevel();

    Vector<ScopedEvent> m_queuedEvents;
    unsigned m_scopingLevel { 0 };

    friend class WTF::NeverDestroyed<WebCore::ScopedEventQueue>;
    friend class EventQueueScope;
};

class EventQueueScope {
    WTF_MAKE_NONCOPYABLE(EventQueueScope);
public:
    EventQueueScope() { ScopedEventQueue::singleton().incrementScopingLevel(); }
    ~EventQueueScope() { ScopedEventQueue::singleton().decrementScopingLevel(); }
};

} // namespace WebCore
