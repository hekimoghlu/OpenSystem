/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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

#include "InspectorAgentRegistry.h"
#include "InspectorEnvironment.h"
#include "InspectorFrontendRouter.h"
#include "Strong.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

#if ENABLE(INSPECTOR_ALTERNATE_DISPATCHERS)
#include "AugmentableInspectorController.h"
#endif

namespace JSC {
class CallFrame;
class ConsoleClient;
class Exception;
class JSGlobalObject;
}

namespace Inspector {

class BackendDispatcher;
class FrontendChannel;
class InjectedScriptManager;
class InspectorAgent;
class InspectorConsoleAgent;
class InspectorDebuggerAgent;
class InspectorScriptProfilerAgent;
class JSGlobalObjectConsoleClient;
class JSGlobalObjectDebugger;
class ScriptCallStack;
struct JSAgentContext;

class JSGlobalObjectInspectorController final
    : public InspectorEnvironment
#if ENABLE(INSPECTOR_ALTERNATE_DISPATCHERS)
    , public AugmentableInspectorController
#endif
{
    WTF_MAKE_NONCOPYABLE(JSGlobalObjectInspectorController);
    WTF_MAKE_TZONE_ALLOCATED(JSGlobalObjectInspectorController);
public:
    JSGlobalObjectInspectorController(JSC::JSGlobalObject&);
    ~JSGlobalObjectInspectorController() final;

    void connectFrontend(FrontendChannel&, bool isAutomaticInspection, bool immediatelyPause);
    void disconnectFrontend(FrontendChannel&);

    void dispatchMessageFromFrontend(const String&);

    void globalObjectDestroyed();

    bool includesNativeCallStackWhenReportingExceptions() const { return m_includeNativeCallStackWithExceptions; }
    void setIncludesNativeCallStackWhenReportingExceptions(bool includesNativeCallStack) { m_includeNativeCallStackWithExceptions = includesNativeCallStack; }

    void reportAPIException(JSC::JSGlobalObject*, JSC::Exception*);

    WeakPtr<JSC::ConsoleClient> consoleClient() const;

    bool developerExtrasEnabled() const final;
    bool canAccessInspectedScriptState(JSC::JSGlobalObject*) const final { return true; }
    InspectorFunctionCallHandler functionCallHandler() const final;
    InspectorEvaluateHandler evaluateHandler() const final;
    void frontendInitialized() final;
    WTF::Stopwatch& executionStopwatch() const final;
    JSC::Debugger* debugger() final;
    JSC::VM& vm() final;

#if ENABLE(INSPECTOR_ALTERNATE_DISPATCHERS)
    AugmentableInspectorControllerClient* augmentableInspectorControllerClient() const final { return m_augmentingClient; } 
    void setAugmentableInspectorControllerClient(AugmentableInspectorControllerClient* client) final { m_augmentingClient = client; }

    const FrontendRouter& frontendRouter() const final { return m_frontendRouter.get(); }
    BackendDispatcher& backendDispatcher() final { return m_backendDispatcher.get(); }
    void registerAlternateAgent(std::unique_ptr<InspectorAgentBase>) final;
#endif

private:
    void appendAPIBacktrace(ScriptCallStack&);

    InspectorAgent& ensureInspectorAgent();
    InspectorDebuggerAgent& ensureDebuggerAgent();

    JSAgentContext jsAgentContext();
    void createLazyAgents();

    JSC::JSGlobalObject& m_globalObject;
    std::unique_ptr<InjectedScriptManager> m_injectedScriptManager;
    std::unique_ptr<JSGlobalObjectConsoleClient> m_consoleClient;
    Ref<WTF::Stopwatch> m_executionStopwatch;
    std::unique_ptr<JSGlobalObjectDebugger> m_debugger;

    AgentRegistry m_agents;
    InspectorConsoleAgent* m_consoleAgent { nullptr };

    // Lazy, but also on-demand agents.
    InspectorAgent* m_inspectorAgent { nullptr };
    InspectorDebuggerAgent* m_debuggerAgent { nullptr };

    Ref<FrontendRouter> m_frontendRouter;
    Ref<BackendDispatcher> m_backendDispatcher;

    // Used to keep the JSGlobalObject and VM alive while we are debugging it.
    JSC::Strong<JSC::JSGlobalObject> m_strongGlobalObject;
    RefPtr<JSC::VM> m_strongVM;

    bool m_includeNativeCallStackWithExceptions { true };
    bool m_isAutomaticInspection { false };
    bool m_pauseAfterInitialization { false };
    bool m_didCreateLazyAgents { false };

#if ENABLE(INSPECTOR_ALTERNATE_DISPATCHERS)
    AugmentableInspectorControllerClient* m_augmentingClient { nullptr };
#endif
};

} // namespace Inspector
