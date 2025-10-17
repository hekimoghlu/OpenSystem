/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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

#if ENABLE(RESOURCE_USAGE)

#include "InspectorWebAgentBase.h"
#include "ResourceUsageData.h"
#include <JavaScriptCore/InspectorBackendDispatchers.h>
#include <JavaScriptCore/InspectorFrontendDispatchers.h>
#include <wtf/MemoryPressureHandler.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class InspectorMemoryAgent final : public InspectorAgentBase, public Inspector::MemoryBackendDispatcherHandler {
    WTF_MAKE_NONCOPYABLE(InspectorMemoryAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorMemoryAgent);
public:
    InspectorMemoryAgent(PageAgentContext&);
    ~InspectorMemoryAgent();

    // InspectorAgentBase
    void didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*);
    void willDestroyFrontendAndBackend(Inspector::DisconnectReason);

    // MemoryBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<void> enable();
    Inspector::Protocol::ErrorStringOr<void> disable();
    Inspector::Protocol::ErrorStringOr<void> startTracking();
    Inspector::Protocol::ErrorStringOr<void> stopTracking();

    // InspectorInstrumentation
    void didHandleMemoryPressure(Critical);

private:
    void collectSample(const ResourceUsageData&);

    std::unique_ptr<Inspector::MemoryFrontendDispatcher> m_frontendDispatcher;
    RefPtr<Inspector::MemoryBackendDispatcher> m_backendDispatcher;
    bool m_tracking { false };
};

} // namespace WebCore

#endif // ENABLE(RESOURCE_USAGE)
