/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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

#include "ApplicationCacheHost.h"
#include "InspectorWebAgentBase.h"
#include <JavaScriptCore/InspectorBackendDispatchers.h>
#include <JavaScriptCore/InspectorFrontendDispatchers.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace Inspector {
class ApplicationCacheFrontendDispatcher;
}

namespace WebCore {

class LocalFrame;
class Page;

class InspectorApplicationCacheAgent final : public InspectorAgentBase, public Inspector::ApplicationCacheBackendDispatcherHandler {
    WTF_MAKE_NONCOPYABLE(InspectorApplicationCacheAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorApplicationCacheAgent);
public:
    InspectorApplicationCacheAgent(PageAgentContext&);
    ~InspectorApplicationCacheAgent();

    // InspectorAgentBase
    void didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*);
    void willDestroyFrontendAndBackend(Inspector::DisconnectReason);

    // ApplicationCacheBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<void> enable();
    Inspector::Protocol::ErrorStringOr<void> disable();
    Inspector::Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Inspector::Protocol::ApplicationCache::FrameWithManifest>>> getFramesWithManifests();
    Inspector::Protocol::ErrorStringOr<String> getManifestForFrame(const Inspector::Protocol::Network::FrameId&);
    Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::ApplicationCache::ApplicationCache>> getApplicationCacheForFrame(const Inspector::Protocol::Network::FrameId&);

    // InspectorInstrumentation
    void updateApplicationCacheStatus(LocalFrame*);
    void networkStateChanged();

private:
    Ref<Inspector::Protocol::ApplicationCache::ApplicationCache> buildObjectForApplicationCache(const Vector<ApplicationCacheHost::ResourceInfo>&, const ApplicationCacheHost::CacheInfo&);
    Ref<JSON::ArrayOf<Inspector::Protocol::ApplicationCache::ApplicationCacheResource>> buildArrayForApplicationCacheResources(const Vector<ApplicationCacheHost::ResourceInfo>&);
    Ref<Inspector::Protocol::ApplicationCache::ApplicationCacheResource> buildObjectForApplicationCacheResource(const ApplicationCacheHost::ResourceInfo&);

    DocumentLoader* assertFrameWithDocumentLoader(Inspector::Protocol::ErrorString&, const Inspector::Protocol::Network::FrameId&);

    std::unique_ptr<Inspector::ApplicationCacheFrontendDispatcher> m_frontendDispatcher;
    RefPtr<Inspector::ApplicationCacheBackendDispatcher> m_backendDispatcher;
    WeakRef<Page> m_inspectedPage;
};

} // namespace WebCore

