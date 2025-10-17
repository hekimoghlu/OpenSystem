/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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

#include "WorkerOrWorkletGlobalScope.h"
#include <JavaScriptCore/InspectorAgentRegistry.h>
#include <JavaScriptCore/InspectorEnvironment.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/Stopwatch.h>
#include <wtf/TZoneMalloc.h>

namespace Inspector {
class FrontendChannel;
class FrontendRouter;
};

namespace WebCore {

class InstrumentingAgents;
class WebInjectedScriptManager;
class WorkerDebugger;
struct WorkerAgentContext;

class WorkerInspectorController final : public Inspector::InspectorEnvironment {
    WTF_MAKE_NONCOPYABLE(WorkerInspectorController);
    WTF_MAKE_TZONE_ALLOCATED(WorkerInspectorController);
public:
    explicit WorkerInspectorController(WorkerOrWorkletGlobalScope&);
    ~WorkerInspectorController() override;

    void workerTerminating();

    void connectFrontend();
    void disconnectFrontend(Inspector::DisconnectReason);

    void dispatchMessageFromFrontend(const String&);

    // InspectorEnvironment
    bool developerExtrasEnabled() const override { return true; }
    bool canAccessInspectedScriptState(JSC::JSGlobalObject*) const override { return true; }
    Inspector::InspectorFunctionCallHandler functionCallHandler() const override;
    Inspector::InspectorEvaluateHandler evaluateHandler() const override;
    void frontendInitialized() override { }
    WTF::Stopwatch& executionStopwatch() const override;
    JSC::Debugger* debugger() override;
    JSC::VM& vm() override;

private:
    friend class InspectorInstrumentation;

    WorkerAgentContext workerAgentContext();
    void createLazyAgents();

    void updateServiceWorkerPageFrontendCount();

    Ref<InstrumentingAgents> m_instrumentingAgents;
    std::unique_ptr<WebInjectedScriptManager> m_injectedScriptManager;
    Ref<Inspector::FrontendRouter> m_frontendRouter;
    Ref<Inspector::BackendDispatcher> m_backendDispatcher;
    Ref<WTF::Stopwatch> m_executionStopwatch;
    std::unique_ptr<WorkerDebugger> m_debugger;
    Inspector::AgentRegistry m_agents;
    WeakRef<WorkerOrWorkletGlobalScope> m_globalScope;
    std::unique_ptr<Inspector::FrontendChannel> m_forwardingChannel;
    bool m_didCreateLazyAgents { false };
};

} // namespace WebCore
