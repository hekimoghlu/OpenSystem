/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
#include "InspectorWorkerAgent.h"

#include "Document.h"
#include "InstrumentingAgents.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorWorkerAgent);

InspectorWorkerAgent::InspectorWorkerAgent(WebAgentContext& context)
    : InspectorAgentBase("Worker"_s, context)
    , m_pageChannel(PageChannel::create(*this))
    , m_frontendDispatcher(makeUniqueRef<Inspector::WorkerFrontendDispatcher>(context.frontendRouter))
    , m_backendDispatcher(Inspector::WorkerBackendDispatcher::create(context.backendDispatcher, this))
{
}

InspectorWorkerAgent::~InspectorWorkerAgent()
{
    m_pageChannel->detachFromParentAgent();
}

void InspectorWorkerAgent::didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*)
{
    m_instrumentingAgents.setPersistentWorkerAgent(this);
}

void InspectorWorkerAgent::willDestroyFrontendAndBackend(DisconnectReason)
{
    m_instrumentingAgents.setPersistentWorkerAgent(nullptr);

    disable();
}

Inspector::Protocol::ErrorStringOr<void> InspectorWorkerAgent::enable()
{
    if (m_enabled)
        return { };

    m_enabled = true;

    connectToAllWorkerInspectorProxies();

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorWorkerAgent::disable()
{
    if (!m_enabled)
        return { };

    m_enabled = false;

    disconnectFromAllWorkerInspectorProxies();

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorWorkerAgent::initialized(const String& workerId)
{
    RefPtr proxy = m_connectedProxies.get(workerId).get();
    if (!proxy)
        return makeUnexpected("Missing worker for given workerId"_s);

    proxy->resumeWorkerIfPaused();

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorWorkerAgent::sendMessageToWorker(const String& workerId, const String& message)
{
    if (!m_enabled)
        return makeUnexpected("Worker domain must be enabled"_s);

    RefPtr proxy = m_connectedProxies.get(workerId).get();
    if (!proxy)
        return makeUnexpected("Missing worker for given workerId"_s);

    proxy->sendMessageToWorkerInspectorController(message);

    return { };
}

bool InspectorWorkerAgent::shouldWaitForDebuggerOnStart() const
{
    return m_enabled;
}

void InspectorWorkerAgent::workerStarted(WorkerInspectorProxy& proxy)
{
    if (!m_enabled)
        return;

    connectToWorkerInspectorProxy(proxy);
}

void InspectorWorkerAgent::workerTerminated(WorkerInspectorProxy& proxy)
{
    if (!m_enabled)
        return;

    disconnectFromWorkerInspectorProxy(proxy);
}

void InspectorWorkerAgent::disconnectFromAllWorkerInspectorProxies()
{
    for (auto& proxyWeakPtr : copyToVector(m_connectedProxies.values())) {
        RefPtr proxy = proxyWeakPtr.get();
        if (!proxy)
            continue;

        proxy->disconnectFromWorkerInspectorController();
    }

    m_connectedProxies.clear();
}

void InspectorWorkerAgent::connectToWorkerInspectorProxy(WorkerInspectorProxy& proxy)
{
    proxy.connectToWorkerInspectorController(m_pageChannel);

    m_connectedProxies.set(proxy.identifier(), proxy);

    m_frontendDispatcher->workerCreated(proxy.identifier(), proxy.url().string(), proxy.name());
}

void InspectorWorkerAgent::disconnectFromWorkerInspectorProxy(WorkerInspectorProxy& proxy)
{
    m_frontendDispatcher->workerTerminated(proxy.identifier());

    m_connectedProxies.remove(proxy.identifier());

    proxy.disconnectFromWorkerInspectorController();
}

Ref<InspectorWorkerAgent::PageChannel> InspectorWorkerAgent::PageChannel::create(InspectorWorkerAgent& parentAgent)
{
    return adoptRef(*new PageChannel(parentAgent));
}

InspectorWorkerAgent::PageChannel::PageChannel(InspectorWorkerAgent& parentAgent)
    : m_parentAgent(&parentAgent)
{
}

void InspectorWorkerAgent::PageChannel::detachFromParentAgent()
{
    Locker locker { m_parentAgentLock };

    m_parentAgent = nullptr;
}

void InspectorWorkerAgent::PageChannel::sendMessageFromWorkerToFrontend(WorkerInspectorProxy& proxy, String&& message)
{
    Locker locker { m_parentAgentLock };

    if (CheckedPtr parentAgent = m_parentAgent)
        parentAgent->frontendDispatcher().dispatchMessageFromWorker(proxy.identifier(), WTFMove(message));
}

} // namespace Inspector
