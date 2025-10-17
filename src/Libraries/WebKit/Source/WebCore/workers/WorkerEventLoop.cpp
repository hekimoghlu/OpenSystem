/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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
#include "WorkerEventLoop.h"

#include "ContextDestructionObserverInlines.h"
#include "Microtasks.h"
#include "WorkerOrWorkletGlobalScope.h"

namespace WebCore {

Ref<WorkerEventLoop> WorkerEventLoop::create(WorkerOrWorkletGlobalScope& context)
{
    return adoptRef(*new WorkerEventLoop(context));
}

WorkerEventLoop::WorkerEventLoop(WorkerOrWorkletGlobalScope& context)
    : ContextDestructionObserver(&context)
{
    addAssociatedContext(context);
}

WorkerEventLoop::~WorkerEventLoop()
{
}

void WorkerEventLoop::scheduleToRun()
{
    auto* globalScope = downcast<WorkerOrWorkletGlobalScope>(scriptExecutionContext());
    ASSERT(globalScope);
    // Post this task with a special event mode, so that it can be separated from other
    // kinds of tasks so that queued microtasks can run even if other tasks are ignored.
    globalScope->postTaskForMode([eventLoop = Ref { *this }] (ScriptExecutionContext&) {
        eventLoop->run();
    }, WorkerEventLoop::taskMode());
}

bool WorkerEventLoop::isContextThread() const
{
    return scriptExecutionContext()->isContextThread();
}

MicrotaskQueue& WorkerEventLoop::microtaskQueue()
{
    ASSERT(scriptExecutionContext());
    if (!m_microtaskQueue)
        m_microtaskQueue = makeUnique<MicrotaskQueue>(scriptExecutionContext()->vm(), *this);
    return *m_microtaskQueue;
}

void WorkerEventLoop::clearMicrotaskQueue()
{
    m_microtaskQueue = nullptr;
}

const String WorkerEventLoop::taskMode()
{
    return "workerEventLoopTaskMode"_s;
}

} // namespace WebCore
