/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
#include "InspectorCPUProfilerAgent.h"

#if ENABLE(RESOURCE_USAGE)

#include "InstrumentingAgents.h"
#include "ResourceUsageThread.h"
#include <JavaScriptCore/InspectorEnvironment.h>
#include <wtf/Stopwatch.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorCPUProfilerAgent);

InspectorCPUProfilerAgent::InspectorCPUProfilerAgent(PageAgentContext& context)
    : InspectorAgentBase("CPUProfiler"_s, context)
    , m_frontendDispatcher(makeUnique<Inspector::CPUProfilerFrontendDispatcher>(context.frontendRouter))
    , m_backendDispatcher(Inspector::CPUProfilerBackendDispatcher::create(context.backendDispatcher, this))
{
}

InspectorCPUProfilerAgent::~InspectorCPUProfilerAgent() = default;

void InspectorCPUProfilerAgent::didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*)
{
    m_instrumentingAgents.setPersistentCPUProfilerAgent(this);
}

void InspectorCPUProfilerAgent::willDestroyFrontendAndBackend(DisconnectReason)
{
    stopTracking();

    m_instrumentingAgents.setPersistentCPUProfilerAgent(nullptr);
}

Inspector::Protocol::ErrorStringOr<void> InspectorCPUProfilerAgent::startTracking()
{
    if (m_tracking)
        return { };

    ResourceUsageThread::addObserver(this, CPU, [this] (const ResourceUsageData& data) {
        collectSample(data);
    });

    m_tracking = true;

    m_frontendDispatcher->trackingStart(m_environment.executionStopwatch().elapsedTime().seconds());

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorCPUProfilerAgent::stopTracking()
{
    if (!m_tracking)
        return { };

    ResourceUsageThread::removeObserver(this);

    m_tracking = false;

    m_frontendDispatcher->trackingComplete(m_environment.executionStopwatch().elapsedTime().seconds());

    return { };
}

static Ref<Inspector::Protocol::CPUProfiler::ThreadInfo> buildThreadInfo(const ThreadCPUInfo& thread)
{
    ASSERT(thread.cpu <= 100);

    auto threadInfo = Inspector::Protocol::CPUProfiler::ThreadInfo::create()
        .setName(thread.name)
        .setUsage(thread.cpu)
        .release();

    if (thread.type == ThreadCPUInfo::Type::Main)
        threadInfo->setType(Inspector::Protocol::CPUProfiler::ThreadInfo::Type::Main);
    else if (thread.type == ThreadCPUInfo::Type::WebKit)
        threadInfo->setType(Inspector::Protocol::CPUProfiler::ThreadInfo::Type::WebKit);

    if (!thread.identifier.isEmpty())
        threadInfo->setTargetId(thread.identifier);

    return threadInfo;
}

void InspectorCPUProfilerAgent::collectSample(const ResourceUsageData& data)
{
    auto event = Inspector::Protocol::CPUProfiler::Event::create()
        .setTimestamp(m_environment.executionStopwatch().elapsedTimeSince(data.timestamp).seconds())
        .setUsage(data.cpuExcludingDebuggerThreads)
        .release();

    if (!data.cpuThreads.isEmpty()) {
        auto threads = JSON::ArrayOf<Inspector::Protocol::CPUProfiler::ThreadInfo>::create();
        for (auto& threadInfo : data.cpuThreads)
            threads->addItem(buildThreadInfo(threadInfo));
        event->setThreads(WTFMove(threads));
    }

    m_frontendDispatcher->trackingUpdate(WTFMove(event));
}

} // namespace WebCore

#endif // ENABLE(RESOURCE_USAGE)
