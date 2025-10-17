/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
#include "InspectorApplicationCacheAgent.h"

#include "ApplicationCacheHost.h"
#include "DocumentLoader.h"
#include "FrameLoader.h"
#include "InspectorPageAgent.h"
#include "InstrumentingAgents.h"
#include "LoaderStrategy.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PlatformStrategies.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorApplicationCacheAgent);

InspectorApplicationCacheAgent::InspectorApplicationCacheAgent(PageAgentContext& context)
    : InspectorAgentBase("ApplicationCache"_s, context)
    , m_frontendDispatcher(makeUnique<Inspector::ApplicationCacheFrontendDispatcher>(context.frontendRouter))
    , m_backendDispatcher(Inspector::ApplicationCacheBackendDispatcher::create(context.backendDispatcher, this))
    , m_inspectedPage(context.inspectedPage)
{
}

InspectorApplicationCacheAgent::~InspectorApplicationCacheAgent() = default;

void InspectorApplicationCacheAgent::didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*)
{
}

void InspectorApplicationCacheAgent::willDestroyFrontendAndBackend(Inspector::DisconnectReason)
{
    disable();
}

Inspector::Protocol::ErrorStringOr<void> InspectorApplicationCacheAgent::enable()
{
    if (m_instrumentingAgents.enabledApplicationCacheAgent() == this)
        return makeUnexpected("ApplicationCache domain already enabled"_s);

    m_instrumentingAgents.setEnabledApplicationCacheAgent(this);

    // We need to pass initial navigator.onOnline.
    networkStateChanged();

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorApplicationCacheAgent::disable()
{
    if (m_instrumentingAgents.enabledApplicationCacheAgent() != this)
        return makeUnexpected("ApplicationCache domain already disabled"_s);

    m_instrumentingAgents.setEnabledApplicationCacheAgent(nullptr);

    return { };
}

void InspectorApplicationCacheAgent::updateApplicationCacheStatus(LocalFrame* frame)
{
    auto* pageAgent = m_instrumentingAgents.enabledPageAgent();
    if (!pageAgent)
        return;

    if (!frame)
        return;

    auto* documentLoader = frame->loader().documentLoader();
    if (!documentLoader)
        return;

    auto& host = documentLoader->applicationCacheHost();
    int status = host.status();
    auto manifestURL = host.applicationCacheInfo().manifest.string();

    m_frontendDispatcher->applicationCacheStatusUpdated(pageAgent->frameId(frame), manifestURL, status);
}

void InspectorApplicationCacheAgent::networkStateChanged()
{
    m_frontendDispatcher->networkStateUpdated(platformStrategies()->loaderStrategy()->isOnLine());
}

Expected<Ref<JSON::ArrayOf<Inspector::Protocol::ApplicationCache::FrameWithManifest>>, Inspector::Protocol::ErrorString> InspectorApplicationCacheAgent::getFramesWithManifests()
{
    auto* pageAgent = m_instrumentingAgents.enabledPageAgent();
    if (!pageAgent)
        return makeUnexpected("Page domain must be enabled"_s);

    auto result = JSON::ArrayOf<Inspector::Protocol::ApplicationCache::FrameWithManifest>::create();
    m_inspectedPage->forEachLocalFrame([&](LocalFrame& frame) {
        auto* documentLoader = frame.loader().documentLoader();
        if (!documentLoader)
            return;

        auto& host = documentLoader->applicationCacheHost();
        String manifestURL = host.applicationCacheInfo().manifest.string();
        if (!manifestURL.isEmpty()) {
            result->addItem(Inspector::Protocol::ApplicationCache::FrameWithManifest::create()
                .setFrameId(pageAgent->frameId(&frame))
                .setManifestURL(manifestURL)
                .setStatus(static_cast<int>(host.status()))
                .release());
        }
    });
    return result;
}

DocumentLoader* InspectorApplicationCacheAgent::assertFrameWithDocumentLoader(Inspector::Protocol::ErrorString& errorString, const Inspector::Protocol::Network::FrameId& frameId)
{
    auto* pageAgent = m_instrumentingAgents.enabledPageAgent();
    if (!pageAgent) {
        errorString = "Page domain must be enabled"_s;
        return nullptr;
    }

    auto* frame = pageAgent->assertFrame(errorString, frameId);
    if (!frame)
        return nullptr;

    return InspectorPageAgent::assertDocumentLoader(errorString, frame);
}

Expected<String, Inspector::Protocol::ErrorString> InspectorApplicationCacheAgent::getManifestForFrame(const Inspector::Protocol::Network::FrameId& frameId)
{
    Inspector::Protocol::ErrorString errorString;

    DocumentLoader* documentLoader = assertFrameWithDocumentLoader(errorString, frameId);
    if (!documentLoader)
        return makeUnexpected(errorString);

    return documentLoader->applicationCacheHost().applicationCacheInfo().manifest.string();
}

Expected<Ref<Inspector::Protocol::ApplicationCache::ApplicationCache>, Inspector::Protocol::ErrorString> InspectorApplicationCacheAgent::getApplicationCacheForFrame(const Inspector::Protocol::Network::FrameId& frameId)
{
    Inspector::Protocol::ErrorString errorString;

    auto* documentLoader = assertFrameWithDocumentLoader(errorString, frameId);
    if (!documentLoader)
        return makeUnexpected(errorString);

    auto& host = documentLoader->applicationCacheHost();
    return buildObjectForApplicationCache(host.resourceList(), host.applicationCacheInfo());
}

Ref<Inspector::Protocol::ApplicationCache::ApplicationCache> InspectorApplicationCacheAgent::buildObjectForApplicationCache(const Vector<ApplicationCacheHost::ResourceInfo>& applicationCacheResources, const ApplicationCacheHost::CacheInfo& applicationCacheInfo)
{
    return Inspector::Protocol::ApplicationCache::ApplicationCache::create()
        .setManifestURL(applicationCacheInfo.manifest.string())
        .setSize(applicationCacheInfo.size)
        .setCreationTime(applicationCacheInfo.creationTime)
        .setUpdateTime(applicationCacheInfo.updateTime)
        .setResources(buildArrayForApplicationCacheResources(applicationCacheResources))
        .release();
}

Ref<JSON::ArrayOf<Inspector::Protocol::ApplicationCache::ApplicationCacheResource>> InspectorApplicationCacheAgent::buildArrayForApplicationCacheResources(const Vector<ApplicationCacheHost::ResourceInfo>& applicationCacheResources)
{
    auto result = JSON::ArrayOf<Inspector::Protocol::ApplicationCache::ApplicationCacheResource>::create();
    for (auto& info : applicationCacheResources)
        result->addItem(buildObjectForApplicationCacheResource(info));
    return result;
}

Ref<Inspector::Protocol::ApplicationCache::ApplicationCacheResource> InspectorApplicationCacheAgent::buildObjectForApplicationCacheResource(const ApplicationCacheHost::ResourceInfo& resourceInfo)
{
    auto types = makeString(resourceInfo.isMaster ? "Master "_s : ""_s,
        resourceInfo.isManifest ? "Manifest "_s : ""_s,
        resourceInfo.isFallback ? "Fallback "_s : ""_s,
        resourceInfo.isForeign ? "Foreign "_s : ""_s,
        resourceInfo.isExplicit ? "Explicit "_s : ""_s);
    return Inspector::Protocol::ApplicationCache::ApplicationCacheResource::create()
        .setUrl(resourceInfo.resource.string())
        .setSize(static_cast<int>(resourceInfo.size))
        .setType(types)
        .release();
}

} // namespace WebCore
