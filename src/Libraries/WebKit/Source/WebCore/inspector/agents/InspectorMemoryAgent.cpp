/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#include "InspectorMemoryAgent.h"

#if ENABLE(RESOURCE_USAGE)

#include "InstrumentingAgents.h"
#include "ResourceUsageThread.h"
#include <JavaScriptCore/InspectorEnvironment.h>
#include <wtf/Stopwatch.h>
#include <wtf/TZoneMallocInlines.h>


namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorMemoryAgent);

InspectorMemoryAgent::InspectorMemoryAgent(PageAgentContext& context)
    : InspectorAgentBase("Memory"_s, context)
    , m_frontendDispatcher(makeUnique<Inspector::MemoryFrontendDispatcher>(context.frontendRouter))
    , m_backendDispatcher(Inspector::MemoryBackendDispatcher::create(context.backendDispatcher, this))
{
}

InspectorMemoryAgent::~InspectorMemoryAgent() = default;

void InspectorMemoryAgent::didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*)
{
    m_instrumentingAgents.setPersistentMemoryAgent(this);
}

void InspectorMemoryAgent::willDestroyFrontendAndBackend(DisconnectReason)
{
    disable();

    m_instrumentingAgents.setPersistentMemoryAgent(nullptr);
}

Inspector::Protocol::ErrorStringOr<void> InspectorMemoryAgent::enable()
{
    if (m_instrumentingAgents.enabledMemoryAgent() == this)
        return makeUnexpected("Memory domain already enabled"_s);

    m_instrumentingAgents.setEnabledMemoryAgent(this);

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorMemoryAgent::disable()
{
    if (m_instrumentingAgents.enabledMemoryAgent() != this)
        return makeUnexpected("Memory domain already disabled"_s);

    m_instrumentingAgents.setEnabledMemoryAgent(nullptr);

    m_tracking = false;

    ResourceUsageThread::removeObserver(this);

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorMemoryAgent::startTracking()
{
    if (m_tracking)
        return { };

    ResourceUsageThread::addObserver(this, Memory, [this] (const ResourceUsageData& data) {
        collectSample(data);
    });

    m_tracking = true;

    m_frontendDispatcher->trackingStart(m_environment.executionStopwatch().elapsedTime().seconds());

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorMemoryAgent::stopTracking()
{
    if (!m_tracking)
        return { };

    ResourceUsageThread::removeObserver(this);

    m_tracking = false;

    m_frontendDispatcher->trackingComplete(m_environment.executionStopwatch().elapsedTime().seconds());

    return { };
}

void InspectorMemoryAgent::didHandleMemoryPressure(Critical critical)
{
    MemoryFrontendDispatcher::Severity severity = critical == Critical::Yes ? MemoryFrontendDispatcher::Severity::Critical : MemoryFrontendDispatcher::Severity::NonCritical;
    m_frontendDispatcher->memoryPressure(m_environment.executionStopwatch().elapsedTime().seconds(), Inspector::Protocol::Helpers::getEnumConstantValue(severity));
}

void InspectorMemoryAgent::collectSample(const ResourceUsageData& data)
{
    auto javascriptCategory = Inspector::Protocol::Memory::CategoryData::create()
        .setType(Inspector::Protocol::Memory::CategoryData::Type::JavaScript)
        .setSize(data.categories[MemoryCategory::GCHeap].totalSize() + data.categories[MemoryCategory::GCOwned].totalSize())
        .release();

    auto jitCategory = Inspector::Protocol::Memory::CategoryData::create()
        .setType(Inspector::Protocol::Memory::CategoryData::Type::JIT)
        .setSize(data.categories[MemoryCategory::JSJIT].totalSize())
        .release();

    auto imagesCategory = Inspector::Protocol::Memory::CategoryData::create()
        .setType(Inspector::Protocol::Memory::CategoryData::Type::Images)
        .setSize(data.categories[MemoryCategory::Images].totalSize())
        .release();

    auto layersCategory = Inspector::Protocol::Memory::CategoryData::create()
        .setType(Inspector::Protocol::Memory::CategoryData::Type::Layers)
        .setSize(data.categories[MemoryCategory::Layers].totalSize())
        .release();

    auto pageCategory = Inspector::Protocol::Memory::CategoryData::create()
        .setType(Inspector::Protocol::Memory::CategoryData::Type::Page)
        .setSize(data.categories[MemoryCategory::bmalloc].totalSize() + data.categories[MemoryCategory::LibcMalloc].totalSize())
        .release();

    auto otherCategory = Inspector::Protocol::Memory::CategoryData::create()
        .setType(Inspector::Protocol::Memory::CategoryData::Type::Other)
        .setSize(data.categories[MemoryCategory::Other].totalSize())
        .release();

    auto categories = JSON::ArrayOf<Inspector::Protocol::Memory::CategoryData>::create();
    categories->addItem(WTFMove(javascriptCategory));
    categories->addItem(WTFMove(jitCategory));
    categories->addItem(WTFMove(imagesCategory));
    categories->addItem(WTFMove(layersCategory));
    categories->addItem(WTFMove(pageCategory));
    categories->addItem(WTFMove(otherCategory));

    auto event = Inspector::Protocol::Memory::Event::create()
        .setTimestamp(m_environment.executionStopwatch().elapsedTimeSince(data.timestamp).seconds())
        .setCategories(WTFMove(categories))
        .release();

    m_frontendDispatcher->trackingUpdate(WTFMove(event));
}

} // namespace WebCore

#endif // ENABLE(RESOURCE_USAGE)
