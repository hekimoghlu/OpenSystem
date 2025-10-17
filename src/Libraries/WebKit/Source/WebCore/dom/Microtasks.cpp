/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#include "Microtasks.h"

#include "CommonVM.h"
#include "EventLoop.h"
#include "JSDOMExceptionHandling.h"
#include "JSExecState.h"
#include "RejectedPromiseTracker.h"
#include "ScriptExecutionContext.h"
#include "WorkerGlobalScope.h"
#include <JavaScriptCore/CatchScope.h>
#include <JavaScriptCore/MicrotaskQueueInlines.h>
#include <JavaScriptCore/VMTrapsInlines.h>
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MicrotaskQueue);
WTF_MAKE_TZONE_ALLOCATED_IMPL(WebCoreMicrotaskDispatcher);


JSC::QueuedTask::Result WebCoreMicrotaskDispatcher::currentRunnability() const
{
    auto group = m_group.get();
    if (!group || group->isStoppedPermanently())
        return JSC::QueuedTask::Result::Discard;
    if (group->isSuspended())
        return JSC::QueuedTask::Result::Suspended;
    return JSC::QueuedTask::Result::Executed;
}

MicrotaskQueue::MicrotaskQueue(JSC::VM& vm, EventLoop& eventLoop)
    : m_vm(vm)
    , m_eventLoop(eventLoop)
    , m_microtaskQueue(vm)
{
}

MicrotaskQueue::~MicrotaskQueue() = default;

void MicrotaskQueue::append(JSC::QueuedTask&& task)
{
    m_microtaskQueue.enqueue(WTFMove(task));
}

void MicrotaskQueue::runJSMicrotask(JSC::JSGlobalObject* globalObject, JSC::VM& vm, JSC::QueuedTask& task)
{
    auto scope = DECLARE_CATCH_SCOPE(vm);

    if (UNLIKELY(!task.job().isObject()))
        return;

    auto* job = JSC::asObject(task.job());

    if (UNLIKELY(!scope.clearExceptionExceptTermination()))
        return;

    auto* lexicalGlobalObject = job->globalObject();
    auto callData = JSC::getCallData(job);
    if (UNLIKELY(!scope.clearExceptionExceptTermination()))
        return;
    ASSERT(callData.type != JSC::CallData::Type::None);

    unsigned count = 0;
    for (auto argument : task.arguments()) {
        if (!argument)
            break;
        ++count;
    }

    if (UNLIKELY(globalObject->hasDebugger())) {
        JSC::DeferTerminationForAWhile deferTerminationForAWhile(vm);
        globalObject->debugger()->willRunMicrotask(globalObject, task.identifier());
        scope.clearException();
    }

    NakedPtr<JSC::Exception> returnedException = nullptr;
    if (LIKELY(!vm.hasPendingTerminationException())) {
        JSC::profiledCall(lexicalGlobalObject, JSC::ProfilingReason::Microtask, job, callData, JSC::jsUndefined(), JSC::ArgList { std::bit_cast<JSC::EncodedJSValue*>(task.arguments().data()), count }, returnedException);
        if (UNLIKELY(returnedException))
            reportException(lexicalGlobalObject, returnedException);
        scope.clearExceptionExceptTermination();
    }

    if (UNLIKELY(globalObject->hasDebugger())) {
        JSC::DeferTerminationForAWhile deferTerminationForAWhile(vm);
        globalObject->debugger()->didRunMicrotask(globalObject, task.identifier());
        scope.clearException();
    }
}

void MicrotaskQueue::performMicrotaskCheckpoint()
{
    if (m_performingMicrotaskCheckpoint)
        return;

    SetForScope change(m_performingMicrotaskCheckpoint, true);
    Ref vm = this->vm();
    JSC::JSLockHolder locker(vm);
    auto catchScope = DECLARE_CATCH_SCOPE(vm);
    {
        SUPPRESS_UNCOUNTED_ARG auto& data = threadGlobalData();
        auto* previousState = data.currentState();
        m_microtaskQueue.performMicrotaskCheckpoint(vm,
            [&](JSC::QueuedTask& task) ALWAYS_INLINE_LAMBDA {
                RefPtr dispatcher = downcast<WebCoreMicrotaskDispatcher>(task.dispatcher());
                if (UNLIKELY(!dispatcher))
                    return JSC::QueuedTask::Result::Discard;

                auto result = dispatcher->currentRunnability();
                if (result == JSC::QueuedTask::Result::Executed) {
                    switch (dispatcher->type()) {
                    case WebCoreMicrotaskDispatcher::Type::JavaScript: {
                        auto* globalObject = task.globalObject();
                        data.setCurrentState(globalObject);
                        runJSMicrotask(globalObject, vm, task);
                        break;
                    }
                    case WebCoreMicrotaskDispatcher::Type::None:
                    case WebCoreMicrotaskDispatcher::Type::UserGestureIndicator:
                    case WebCoreMicrotaskDispatcher::Type::Function:
                        data.setCurrentState(previousState);
                        dispatcher->run(task);
                        break;
                    }
                }
                return result;
            });
        data.setCurrentState(previousState);
    }
    vm->finalizeSynchronousJSExecution();

    if (!vm->executionForbidden()) {
        auto checkpointTasks = std::exchange(m_checkpointTasks, { });
        for (auto& checkpointTask : checkpointTasks) {
            auto* group = checkpointTask->group();
            if (!group || group->isStoppedPermanently())
                continue;

            if (group->isSuspended()) {
                m_checkpointTasks.append(WTFMove(checkpointTask));
                continue;
            }

            checkpointTask->execute();
            if (UNLIKELY(!catchScope.clearExceptionExceptTermination()))
                break; // Encountered termination.
        }
    }

    // https://html.spec.whatwg.org/multipage/webappapis.html#perform-a-microtask-checkpoint (step 4).
    Ref { *m_eventLoop }->forEachAssociatedContext([vm = vm.copyRef()](auto& context) {
        if (UNLIKELY(vm->executionForbidden()))
            return;
        auto catchScope = DECLARE_CATCH_SCOPE(vm);
        if (CheckedPtr tracker = context.rejectedPromiseTracker())
            tracker->processQueueSoon();
        catchScope.clearExceptionExceptTermination();
    });

    // FIXME: We should cleanup Indexed Database transactions as per:
    // https://html.spec.whatwg.org/multipage/webappapis.html#perform-a-microtask-checkpoint (step 5).
}

void MicrotaskQueue::addCheckpointTask(std::unique_ptr<EventLoopTask>&& task)
{
    m_checkpointTasks.append(WTFMove(task));
}

bool MicrotaskQueue::hasMicrotasksForFullyActiveDocument() const
{
    return m_microtaskQueue.hasMicrotasksForFullyActiveDocument();
}

} // namespace WebCore
