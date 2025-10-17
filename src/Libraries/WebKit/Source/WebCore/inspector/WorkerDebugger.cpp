/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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
#include "WorkerDebugger.h"

#include "JSDOMExceptionHandling.h"
#include "Timer.h"
#include "WorkerOrWorkletGlobalScope.h"
#include "WorkerOrWorkletScriptController.h"
#include "WorkerRunLoop.h"
#include "WorkerThread.h"
#include <JavaScriptCore/VM.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WorkerDebugger);

WorkerDebugger::WorkerDebugger(WorkerOrWorkletGlobalScope& context)
    : Debugger(context.script()->vm())
    , m_globalScope(context)
{
}

void WorkerDebugger::attachDebugger()
{
    JSC::Debugger::attachDebugger();

    m_globalScope->script()->attachDebugger(this);
}

void WorkerDebugger::detachDebugger(bool isBeingDestroyed)
{
    JSC::Debugger::detachDebugger(isBeingDestroyed);

    if (m_globalScope->script())
        m_globalScope->script()->detachDebugger(this);
    if (!isBeingDestroyed)
        recompileAllJSFunctions();
}

void WorkerDebugger::recompileAllJSFunctions()
{
    JSC::JSLockHolder lock(vm());
    JSC::Debugger::recompileAllJSFunctions();
}

void WorkerDebugger::runEventLoopWhilePaused()
{
    JSC::Debugger::runEventLoopWhilePaused();

    TimerBase::fireTimersInNestedEventLoop();

    // FIXME: Add support for pausing workers running on the main thread.
    if (!is<WorkerDedicatedRunLoop>(m_globalScope->workerOrWorkletThread()->runLoop()))
        return;

    MessageQueueWaitResult result;
    do {
        result = downcast<WorkerDedicatedRunLoop>(m_globalScope->workerOrWorkletThread()->runLoop()).runInDebuggerMode(m_globalScope);
    } while (result != MessageQueueTerminated && !m_doneProcessingDebuggerEvents);
}

void WorkerDebugger::reportException(JSC::JSGlobalObject* exec, JSC::Exception* exception) const
{
    JSC::Debugger::reportException(exec, exception);

    WebCore::reportException(exec, exception);
}

} // namespace WebCore
