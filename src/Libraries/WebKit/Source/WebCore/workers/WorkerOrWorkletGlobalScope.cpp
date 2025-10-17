/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
#include "WorkerOrWorkletGlobalScope.h"

#include "NoiseInjectionPolicy.h"
#include "ScriptModuleLoader.h"
#include "ServiceWorkerGlobalScope.h"
#include "WorkerEventLoop.h"
#include "WorkerInspectorController.h"
#include "WorkerOrWorkletScriptController.h"
#include "WorkerOrWorkletThread.h"
#include "WorkerRunLoop.h"
#include "WorkletGlobalScope.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WorkerOrWorkletGlobalScope);

WorkerOrWorkletGlobalScope::WorkerOrWorkletGlobalScope(WorkerThreadType type, PAL::SessionID sessionID, Ref<JSC::VM>&& vm, ReferrerPolicy referrerPolicy, WorkerOrWorkletThread* thread, std::optional<uint64_t> noiseInjectionHashSalt, OptionSet<AdvancedPrivacyProtections> advancedPrivacyProtections, std::optional<ScriptExecutionContextIdentifier> contextIdentifier)
    : ScriptExecutionContext(Type::WorkerOrWorkletGlobalScope, contextIdentifier)
    , m_script(makeUnique<WorkerOrWorkletScriptController>(type, WTFMove(vm), this))
    , m_moduleLoader(makeUnique<ScriptModuleLoader>(this, ScriptModuleLoader::OwnerType::WorkerOrWorklet))
    , m_thread(thread)
    , m_inspectorController(makeUnique<WorkerInspectorController>(*this))
    , m_sessionID(sessionID)
    , m_referrerPolicy(referrerPolicy)
    , m_noiseInjectionHashSalt(noiseInjectionHashSalt)
    , m_advancedPrivacyProtections(advancedPrivacyProtections)
{
    relaxAdoptionRequirement();
}

WorkerOrWorkletGlobalScope::~WorkerOrWorkletGlobalScope() = default;

void WorkerOrWorkletGlobalScope::prepareForDestruction()
{
    if (m_defaultTaskGroup) {
        m_defaultTaskGroup->markAsReadyToStop();
        ASSERT(m_defaultTaskGroup->isStoppedPermanently());
    }

    stopActiveDOMObjects();

    // Event listeners would keep DOMWrapperWorld objects alive for too long. Also, they have references to JS objects,
    // which become dangling once Heap is destroyed.
    removeAllEventListeners();

    // MicrotaskQueue and RejectedPromiseTracker reference Heap.
    if (m_eventLoop)
        m_eventLoop->clearMicrotaskQueue();
    removeRejectedPromiseTracker();

    m_inspectorController->workerTerminating();
}

void WorkerOrWorkletGlobalScope::clearScript()
{
    m_script = nullptr;
}

JSC::VM& WorkerOrWorkletGlobalScope::vm()
{
    return script()->vm();
}

JSC::VM* WorkerOrWorkletGlobalScope::vmIfExists() const
{
    return &script()->vm();
}

void WorkerOrWorkletGlobalScope::disableEval(const String& errorMessage)
{
    m_script->disableEval(errorMessage);
}

void WorkerOrWorkletGlobalScope::disableWebAssembly(const String& errorMessage)
{
    m_script->disableWebAssembly(errorMessage);
}

void WorkerOrWorkletGlobalScope::setRequiresTrustedTypes(bool required)
{
    m_script->setRequiresTrustedTypes(required);
}

bool WorkerOrWorkletGlobalScope::isJSExecutionForbidden() const
{
    return !m_script || m_script->isExecutionForbidden();
}

EventLoopTaskGroup& WorkerOrWorkletGlobalScope::eventLoop()
{
    ASSERT(isContextThread());
    if (UNLIKELY(!m_defaultTaskGroup)) {
        m_eventLoop = WorkerEventLoop::create(*this);
        m_defaultTaskGroup = makeUnique<EventLoopTaskGroup>(*m_eventLoop);
        if (activeDOMObjectsAreStopped())
            m_defaultTaskGroup->stopAndDiscardAllTasks();
    }
    return *m_defaultTaskGroup;
}

bool WorkerOrWorkletGlobalScope::isContextThread() const
{
    auto* thread = workerOrWorkletThread();
    return thread && thread->thread() ? thread->thread() == &Thread::current() : isMainThread();
}

void WorkerOrWorkletGlobalScope::postTask(Task&& task)
{
    ASSERT(workerOrWorkletThread());
    workerOrWorkletThread()->runLoop().postTask(WTFMove(task));
}

void WorkerOrWorkletGlobalScope::postTaskForMode(Task&& task, const String& mode)
{
    ASSERT(workerOrWorkletThread());
    workerOrWorkletThread()->runLoop().postTaskForMode(WTFMove(task), mode);
}

OptionSet<NoiseInjectionPolicy> WorkerOrWorkletGlobalScope::noiseInjectionPolicies() const
{
    OptionSet<NoiseInjectionPolicy> policies;
    if (m_advancedPrivacyProtections.contains(AdvancedPrivacyProtections::FingerprintingProtections))
        policies.add(NoiseInjectionPolicy::Minimal);
    if (m_advancedPrivacyProtections.contains(AdvancedPrivacyProtections::ScriptTelemetry))
        policies.add(NoiseInjectionPolicy::Enhanced);
    return policies;
}

} // namespace WebCore
