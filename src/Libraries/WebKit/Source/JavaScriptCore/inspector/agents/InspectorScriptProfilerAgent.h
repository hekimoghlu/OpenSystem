/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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

#include "Debugger.h"
#include "InspectorAgentBase.h"
#include "InspectorBackendDispatchers.h"
#include "InspectorFrontendDispatchers.h"
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {
class Profile;
}

namespace Inspector {

class JS_EXPORT_PRIVATE InspectorScriptProfilerAgent final : public InspectorAgentBase, public ScriptProfilerBackendDispatcherHandler, public JSC::Debugger::ProfilingClient {
    WTF_MAKE_NONCOPYABLE(InspectorScriptProfilerAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorScriptProfilerAgent);
public:
    InspectorScriptProfilerAgent(AgentContext&);
    ~InspectorScriptProfilerAgent() final;

    // InspectorAgentBase
    void didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*) final;
    void willDestroyFrontendAndBackend(DisconnectReason) final;

    // ScriptProfilerBackendDispatcherHandler
    Protocol::ErrorStringOr<void> startTracking(std::optional<bool>&& includeSamples) final;
    Protocol::ErrorStringOr<void> stopTracking() final;

    // JSC::Debugger::ProfilingClient
    bool isAlreadyProfiling() const final;
    Seconds willEvaluateScript() final;
    void didEvaluateScript(Seconds, JSC::ProfilingReason) final;

private:
    void addEvent(Seconds startTime, Seconds endTime, JSC::ProfilingReason);
    void trackingComplete();
    void stopSamplingWhenDisconnecting();

    std::unique_ptr<ScriptProfilerFrontendDispatcher> m_frontendDispatcher;
    RefPtr<ScriptProfilerBackendDispatcher> m_backendDispatcher;
    InspectorEnvironment& m_environment;
    bool m_tracking { false };
#if ENABLE(SAMPLING_PROFILER)
    bool m_enabledSamplingProfiler { false };
#endif
    bool m_activeEvaluateScript { false };
};

} // namespace Inspector
