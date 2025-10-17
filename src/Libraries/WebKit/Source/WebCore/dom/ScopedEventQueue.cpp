/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
#include "ScopedEventQueue.h"

#include "Event.h"
#include "Node.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScopedEventQueue);

ScopedEventQueue& ScopedEventQueue::singleton()
{
    static NeverDestroyed<ScopedEventQueue> scopedEventQueue;
    return scopedEventQueue;
}

void ScopedEventQueue::enqueueEvent(ScopedEvent&& event)
{
    if (m_scopingLevel)
        m_queuedEvents.append(WTFMove(event));
    else
        dispatchEvent(event);
}

void ScopedEventQueue::dispatchEvent(const ScopedEvent& event) const
{
    if (event.event->interfaceType() == EventInterfaceType::MutationEvent && event.target->isInShadowTree())
        return;
    Ref { event.target.get() }->dispatchEvent(event.event);
}

void ScopedEventQueue::dispatchAllEvents()
{
    auto queuedEvents = std::exchange(m_queuedEvents, { });
    for (auto& queuedEvent : queuedEvents)
        dispatchEvent(queuedEvent);
}

void ScopedEventQueue::decrementScopingLevel()
{
    ASSERT(m_scopingLevel);
    --m_scopingLevel;
    if (!m_scopingLevel)
        dispatchAllEvents();
}

}
