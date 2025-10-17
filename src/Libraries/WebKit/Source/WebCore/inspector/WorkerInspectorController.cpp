/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
#include "WorkerInspectorController.h"

#include "CommandLineAPIHost.h"
#include "InspectorClient.h"
#include "InspectorController.h"
#include "InstrumentingAgents.h"
#include "JSExecState.h"
#include "Page.h"
#include "ServiceWorkerAgent.h"
#include "ServiceWorkerGlobalScope.h"
#include "WebHeapAgent.h"
#include "WebInjectedScriptHost.h"
#include "WebInjectedScriptManager.h"
#include "WorkerAuditAgent.h"
#include "WorkerCanvasAgent.h"
#include "WorkerConsoleAgent.h"
#include "WorkerDOMDebuggerAgent.h"
#include "WorkerDebugger.h"
#include "WorkerDebuggerAgent.h"
#include "WorkerNetworkAgent.h"
#include "WorkerOrWorkletGlobalScope.h"
#include "WorkerRuntimeAgent.h"
#include "WorkerThread.h"
#include "WorkerToPageFrontendChannel.h"
#include "WorkerWorkerAgent.h"
#include <JavaScriptCore/InspectorAgentBase.h>
#include <JavaScriptCore/InspectorBackendDispatcher.h>
#include <JavaScriptCore/InspectorFrontendChannel.h>
#include <JavaScriptCore/InspectorFrontendDispatchers.h>
#include <JavaScriptCore/InspectorFrontendRouter.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace JSC;
using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WorkerInspectorController);

WorkerInspectorController::WorkerInspectorController(WorkerOrWorkletGlobalScope& globalScope)
    : m_instrumentingAgents(InstrumentingAgents::create(*this))
    , m_injectedScriptManager(makeUnique<WebInjectedScriptManager>(*this, WebInjectedScriptHost::create()))
    , m_frontendRouter(FrontendRouter::create())
    , m_backendDispatcher(BackendDispatcher::create(m_frontendRouter.copyRef()))
    , m_executionStopwatch(Stopwatch::create())
    , m_globalScope(globalScope)
{
    ASSERT(globalScope.isContextThread());

    auto workerContext = workerAgentContext();

    auto consoleAgent = makeUnique<WorkerConsoleAgent>(workerContext);
    m_instrumentingAgents->setWebConsoleAgent(consoleAgent.get());
    m_agents.append(WTFMove(consoleAgent));
}

WorkerInspectorController::~WorkerInspectorController()
{
    ASSERT(!m_frontendRouter->hasFrontends());
    ASSERT(!m_forwardingChannel);

    m_instrumentingAgents->reset();
}

void WorkerInspectorController::workerTerminating()
{
    m_injectedScriptManager->disconnect();

    disconnectFrontend(Inspector::DisconnectReason::InspectedTargetDestroyed);

    m_agents.discardValues();

    m_debugger = nullptr;
}

void WorkerInspectorController::connectFrontend()
{
    ASSERT(!m_frontendRouter->hasFrontends());
    ASSERT(!m_forwardingChannel);

    createLazyAgents();

    callOnMainThread([] {
        InspectorInstrumentation::frontendCreated();
    });

    m_executionStopwatch->reset();
    m_executionStopwatch->start();

    m_forwardingChannel = makeUnique<WorkerToPageFrontendChannel>(m_globalScope);
    m_frontendRouter->connectFrontend(*m_forwardingChannel.get());
    m_agents.didCreateFrontendAndBackend(&m_frontendRouter.get(), &m_backendDispatcher.get());

    updateServiceWorkerPageFrontendCount();
}

void WorkerInspectorController::disconnectFrontend(Inspector::DisconnectReason reason)
{
    if (!m_frontendRouter->hasFrontends())
        return;

    ASSERT(m_forwardingChannel);

    callOnMainThread([] {
        InspectorInstrumentation::frontendDeleted();
    });

    m_agents.willDestroyFrontendAndBackend(reason);
    m_frontendRouter->disconnectFrontend(*m_forwardingChannel.get());
    m_forwardingChannel = nullptr;

    updateServiceWorkerPageFrontendCount();
}

void WorkerInspectorController::updateServiceWorkerPageFrontendCount()
{
    if (!is<ServiceWorkerGlobalScope>(m_globalScope))
        return;

    auto serviceWorkerPage = downcast<ServiceWorkerGlobalScope>(m_globalScope.get()).serviceWorkerPage();
    if (!serviceWorkerPage)
        return;

    ASSERT(isMainThread());

    // When a service worker is loaded in a Page, we need to report its inspector frontend count
    // up to the page's inspectorController so the client knows about it.
    auto inspectorClient = serviceWorkerPage->inspectorController().inspectorClient();
    if (!inspectorClient)
        return;

    inspectorClient->frontendCountChanged(m_frontendRouter->frontendCount());
}

void WorkerInspectorController::dispatchMessageFromFrontend(const String& message)
{
    m_backendDispatcher->dispatch(message);
}

WorkerAgentContext WorkerInspectorController::workerAgentContext()
{
    AgentContext baseContext = {
        *this,
        *m_injectedScriptManager,
        m_frontendRouter.get(),
        m_backendDispatcher.get(),
    };

    WebAgentContext webContext = {
        baseContext,
        m_instrumentingAgents.get(),
    };

    WorkerAgentContext workerContext = {
        webContext,
        m_globalScope,
    };

    return workerContext;
}

void WorkerInspectorController::createLazyAgents()
{
    if (m_didCreateLazyAgents)
        return;

    m_didCreateLazyAgents = true;

    m_debugger = makeUnique<WorkerDebugger>(m_globalScope);

    m_injectedScriptManager->connect();

    auto workerContext = workerAgentContext();

    m_agents.append(makeUnique<WorkerRuntimeAgent>(workerContext));

    if (is<ServiceWorkerGlobalScope>(m_globalScope)) {
        m_agents.append(makeUnique<ServiceWorkerAgent>(workerContext));
        m_agents.append(makeUnique<WorkerNetworkAgent>(workerContext));
    }

    m_agents.append(makeUnique<WebHeapAgent>(workerContext));

    auto debuggerAgent = makeUnique<WorkerDebuggerAgent>(workerContext);
    auto debuggerAgentPtr = debuggerAgent.get();
    m_agents.append(WTFMove(debuggerAgent));

    m_agents.append(makeUnique<WorkerDOMDebuggerAgent>(workerContext, debuggerAgentPtr));
    m_agents.append(makeUnique<WorkerAuditAgent>(workerContext));
    m_agents.append(makeUnique<WorkerCanvasAgent>(workerContext));
    m_agents.append(makeUnique<WorkerWorkerAgent>(workerContext));

    if (auto& commandLineAPIHost = m_injectedScriptManager->commandLineAPIHost())
        commandLineAPIHost->init(m_instrumentingAgents.copyRef());
}

InspectorFunctionCallHandler WorkerInspectorController::functionCallHandler() const
{
    return WebCore::functionCallHandlerFromAnyThread;
}

InspectorEvaluateHandler WorkerInspectorController::evaluateHandler() const
{
    return WebCore::evaluateHandlerFromAnyThread;
}

Stopwatch& WorkerInspectorController::executionStopwatch() const
{
    return m_executionStopwatch;
}

JSC::Debugger* WorkerInspectorController::debugger()
{
    ASSERT_IMPLIES(m_didCreateLazyAgents, m_debugger);
    return m_debugger.get();
}

VM& WorkerInspectorController::vm()
{
    return m_globalScope->vm();
}

} // namespace WebCore
