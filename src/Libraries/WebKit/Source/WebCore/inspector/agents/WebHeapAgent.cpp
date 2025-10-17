/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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
#include "WebHeapAgent.h"

#include "InstrumentingAgents.h"
#include "WebConsoleAgent.h"
#include <JavaScriptCore/InspectorProtocolTypes.h>
#include <wtf/Lock.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebHeapAgent);

struct GarbageCollectionData {
    Inspector::Protocol::Heap::GarbageCollection::Type type;
    Seconds startTime;
    Seconds endTime;
};

class SendGarbageCollectionEventsTask final : public CanMakeThreadSafeCheckedPtr<SendGarbageCollectionEventsTask> {
    WTF_MAKE_TZONE_ALLOCATED(SendGarbageCollectionEventsTask);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SendGarbageCollectionEventsTask);
public:
    SendGarbageCollectionEventsTask(WebHeapAgent&);
    void addGarbageCollection(GarbageCollectionData&&);
    void reset();
private:
    void timerFired();

    WebHeapAgent& m_agent;
    Lock m_collectionsLock;
    Vector<GarbageCollectionData> m_collections WTF_GUARDED_BY_LOCK(m_collectionsLock);
    RunLoop::Timer m_timer;
};

WTF_MAKE_TZONE_ALLOCATED_IMPL(SendGarbageCollectionEventsTask);

SendGarbageCollectionEventsTask::SendGarbageCollectionEventsTask(WebHeapAgent& agent)
    : m_agent(agent)
    , m_timer(RunLoop::main(), this, &SendGarbageCollectionEventsTask::timerFired)
{
}

void SendGarbageCollectionEventsTask::addGarbageCollection(GarbageCollectionData&& collection)
{
    {
        Locker locker { m_collectionsLock };
        m_collections.append(WTFMove(collection));
    }

    if (!m_timer.isActive())
        m_timer.startOneShot(0_s);
}

void SendGarbageCollectionEventsTask::reset()
{
    {
        Locker locker { m_collectionsLock };
        m_collections.clear();
    }

    m_timer.stop();
}

void SendGarbageCollectionEventsTask::timerFired()
{
    Vector<GarbageCollectionData> collectionsToSend;

    {
        Locker locker { m_collectionsLock };
        m_collections.swap(collectionsToSend);
    }

    m_agent.dispatchGarbageCollectionEventsAfterDelay(WTFMove(collectionsToSend));
}

WebHeapAgent::WebHeapAgent(WebAgentContext& context)
    : InspectorHeapAgent(context)
    , m_instrumentingAgents(context.instrumentingAgents)
    , m_sendGarbageCollectionEventsTask(makeUnique<SendGarbageCollectionEventsTask>(*this))
{
}

WebHeapAgent::~WebHeapAgent() = default;

Inspector::Protocol::ErrorStringOr<void> WebHeapAgent::enable()
{
    auto result = InspectorHeapAgent::enable();

    if (auto* consoleAgent = m_instrumentingAgents.webConsoleAgent())
        consoleAgent->setHeapAgent(this);

    return result;
}

Inspector::Protocol::ErrorStringOr<void> WebHeapAgent::disable()
{
    m_sendGarbageCollectionEventsTask->reset();

    if (auto* consoleAgent = m_instrumentingAgents.webConsoleAgent())
        consoleAgent->setHeapAgent(nullptr);

    return InspectorHeapAgent::disable();
}

void WebHeapAgent::dispatchGarbageCollectedEvent(Inspector::Protocol::Heap::GarbageCollection::Type type, Seconds startTime, Seconds endTime)
{
    // Dispatch the event asynchronously because this method may be
    // called between collection and sweeping and we don't want to
    // create unexpected JavaScript allocations that the Sweeper does
    // not expect to encounter. JavaScript allocations could happen
    // with WebKitLegacy's in process inspector which shares the same
    // VM as the inspected page.

    GarbageCollectionData data = {type, startTime, endTime};
    m_sendGarbageCollectionEventsTask->addGarbageCollection(WTFMove(data));
}

void WebHeapAgent::dispatchGarbageCollectionEventsAfterDelay(Vector<GarbageCollectionData>&& collections)
{
    for (auto& collection : collections)
        InspectorHeapAgent::dispatchGarbageCollectedEvent(collection.type, collection.startTime, collection.endTime);
}

} // namespace WebCore
