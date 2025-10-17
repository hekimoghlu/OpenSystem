/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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
#include "CommonVM.h"

#include "JSDOMWindow.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "OpportunisticTaskScheduler.h"
#include "ScriptController.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/EdenGCActivityCallback.h>
#include <JavaScriptCore/FullGCActivityCallback.h>
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/MachineStackMarker.h>
#include <JavaScriptCore/VM.h>
#include <wtf/MainThread.h>
#include <wtf/RunLoop.h>
#include <wtf/text/AtomString.h>

#if PLATFORM(IOS_FAMILY)
#include "WebCoreThreadInternal.h"
#endif

namespace WebCore {

JSC::VM* g_commonVMOrNull;

JSC::VM& commonVMSlow()
{
    ASSERT(isMainThread());
    ASSERT(!g_commonVMOrNull);

    // FIXME: Remove this call to ScriptController::initializeMainThread(). The
    // main thread should have been initialized by a WebKit entrypoint already.
    // Also, initializeMainThread() does nothing on iOS.
    ScriptController::initializeMainThread();

#if PLATFORM(IOS_FAMILY)
    RunLoop* runLoop = RunLoop::webIfExists();
#else
    RunLoop* runLoop = nullptr;
#endif

    auto& vm = JSC::VM::create(JSC::HeapType::Large, runLoop).leakRef();
#if !PLATFORM(IOS_FAMILY)
    vm.heap.setFullActivityCallback(OpportunisticTaskScheduler::FullGCActivityCallback::create(vm.heap));
    vm.heap.setEdenActivityCallback(OpportunisticTaskScheduler::EdenGCActivityCallback::create(vm.heap));
    vm.heap.disableStopIfNecessaryTimer(); // Because opportunistic task scheduler and GC timer exists, we do not need StopIfNecessaryTimer.
#endif

    g_commonVMOrNull = &vm;

    vm.heap.acquireAccess(); // At any time, we may do things that affect the GC.

#if PLATFORM(IOS_FAMILY)
    if (WebThreadIsEnabled())
        vm.apiLock().makeWebThreadAware();
    vm.heap.machineThreads().addCurrentThread();
#endif

    JSVMClientData::initNormalWorld(&vm, WorkerThreadType::Main);

    return vm;
}

LocalFrame* lexicalFrameFromCommonVM()
{
    JSC::VM& vm = commonVM();
    if (auto* topCallFrame = vm.topCallFrame) {
        if (auto* globalObject = JSC::jsCast<JSDOMGlobalObject*>(topCallFrame->lexicalGlobalObject(vm))) {
            if (auto* window = JSC::jsDynamicCast<JSDOMWindow*>(globalObject)) {
                if (auto* frame = window->wrapped().frame())
                    return dynamicDowncast<LocalFrame>(frame);
            }
        }
    }
    return nullptr;
}

void addImpureProperty(const AtomString& propertyName)
{
    commonVM().addImpureProperty(propertyName.impl());
}

} // namespace WebCore

