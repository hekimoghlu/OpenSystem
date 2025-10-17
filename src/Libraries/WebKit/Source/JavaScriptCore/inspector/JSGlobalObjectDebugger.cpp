/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#include "JSGlobalObjectDebugger.h"

#include "JSGlobalObject.h"
#include "JSLock.h"
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>

namespace Inspector {

using namespace JSC;

WTF_MAKE_TZONE_ALLOCATED_IMPL(JSGlobalObjectDebugger);

JSGlobalObjectDebugger::JSGlobalObjectDebugger(JSGlobalObject& globalObject)
    : Debugger(globalObject.vm())
    , m_globalObject(globalObject)
{
}

void JSGlobalObjectDebugger::attachDebugger()
{
    JSC::Debugger::attachDebugger();

    attach(&m_globalObject);
}

void JSGlobalObjectDebugger::detachDebugger(bool isBeingDestroyed)
{
    JSC::Debugger::detachDebugger(isBeingDestroyed);

    detach(&m_globalObject, isBeingDestroyed ? Debugger::GlobalObjectIsDestructing : Debugger::TerminatingDebuggingSession);
    if (!isBeingDestroyed)
        recompileAllJSFunctions();
}

void JSGlobalObjectDebugger::runEventLoopWhilePaused()
{
    JSC::Debugger::runEventLoopWhilePaused();

    // Drop all locks so another thread can work in the VM while we are nested.
    JSC::JSLock::DropAllLocks dropAllLocks(&m_globalObject.vm());

    while (!m_doneProcessingDebuggerEvents) {
        if (RunLoop::cycle(JSGlobalObjectDebugger::runLoopMode()) == RunLoop::CycleResult::Stop)
            break;
    }
}

RunLoopMode JSGlobalObjectDebugger::runLoopMode()
{
#if USE(CF) && !PLATFORM(WATCHOS)
    // Run the RunLoop in a custom run loop mode to prevent default observers
    // to run and potentially evaluate JavaScript in this context while we are
    // nested. Only the debugger should control things until we continue.
    // FIXME: This is not a perfect solution, as background threads are not
    // paused and can still access and evalute script in the JSContext.

    // FIXME: <rdar://problem/25972777>. On watchOS, in order for auto-attach to work,
    // we need to run in the default run loop mode otherwise we do not receive the XPC messages
    // necessary to setup the relay connection and negotiate an auto-attach debugger.
    return CFSTR("com.apple.JavaScriptCore.remote-inspector-runloop-mode");
#else
    return DefaultRunLoopMode;
#endif
}

} // namespace Inspector
