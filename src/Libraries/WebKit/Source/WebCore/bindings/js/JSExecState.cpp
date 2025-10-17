/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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
#include "JSExecState.h"

#include "EventLoop.h"
#include "JSDOMExceptionHandling.h"
#include "Microtasks.h"
#include "RejectedPromiseTracker.h"
#include "ScriptExecutionContext.h"
#include "WorkerGlobalScope.h"
#include <JavaScriptCore/VMTrapsInlines.h>

namespace WebCore {

void JSExecState::didLeaveScriptContext(JSC::JSGlobalObject* lexicalGlobalObject)
{
    auto context = executionContext(lexicalGlobalObject);
    if (!context)
        return;
    context->eventLoop().performMicrotaskCheckpoint();
}

JSC::JSValue functionCallHandlerFromAnyThread(JSC::JSGlobalObject* lexicalGlobalObject, JSC::JSValue functionObject, const JSC::CallData& callData, JSC::JSValue thisValue, const JSC::ArgList& args, NakedPtr<JSC::Exception>& returnedException)
{
    return JSExecState::call(lexicalGlobalObject, functionObject, callData, thisValue, args, returnedException);
}

JSC::JSValue evaluateHandlerFromAnyThread(JSC::JSGlobalObject* lexicalGlobalObject, const JSC::SourceCode& source, JSC::JSValue thisValue, NakedPtr<JSC::Exception>& returnedException)
{
    return JSExecState::evaluate(lexicalGlobalObject, source, thisValue, returnedException);
}

void JSExecState::runTask(JSC::JSGlobalObject* globalObject, JSC::QueuedTask& task)
{
    JSExecState currentState(globalObject);
    MicrotaskQueue::runJSMicrotask(globalObject, globalObject->vm(), task);
}

ScriptExecutionContext* executionContext(JSC::JSGlobalObject* globalObject)
{
    if (!globalObject || !globalObject->inherits<JSDOMGlobalObject>())
        return nullptr;
    return JSC::jsCast<JSDOMGlobalObject*>(globalObject)->scriptExecutionContext();
}

} // namespace WebCore
