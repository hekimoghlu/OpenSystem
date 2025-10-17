/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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

#include "CustomElementReactionQueue.h"
#include "JSDOMBinding.h"
#include "ThreadGlobalData.h"
#include <JavaScriptCore/CatchScope.h>
#include <JavaScriptCore/Completion.h>
#include <JavaScriptCore/JSMicrotask.h>
#include <JavaScriptCore/Microtask.h>
#include <wtf/ForbidHeapAllocation.h>
#include <wtf/MainThread.h>

namespace WebCore {

class ScriptExecutionContext;

class JSExecState {
    WTF_MAKE_NONCOPYABLE(JSExecState);
    WTF_FORBID_HEAP_ALLOCATION;
    friend class JSMainThreadNullState;
public:
    static JSC::JSGlobalObject* currentState()
    {
        return threadGlobalData().currentState();
    }
    
    static JSC::JSValue call(JSC::JSGlobalObject* lexicalGlobalObject, JSC::JSValue functionObject, const JSC::CallData& callData, JSC::JSValue thisValue, const JSC::ArgList& args, NakedPtr<JSC::Exception>& returnedException)
    {
        JSC::VM& vm = JSC::getVM(lexicalGlobalObject);
        auto scope = DECLARE_CATCH_SCOPE(vm);
        JSC::JSValue returnValue;
        {
            JSExecState currentState(lexicalGlobalObject);
            returnValue = JSC::call(lexicalGlobalObject, functionObject, callData, thisValue, args, returnedException);
        }
        scope.assertNoExceptionExceptTermination();
        return returnValue;
    }

    static JSC::JSValue evaluate(JSC::JSGlobalObject* lexicalGlobalObject, const JSC::SourceCode& source, JSC::JSValue thisValue, NakedPtr<JSC::Exception>& returnedException)
    {
        JSC::VM& vm = JSC::getVM(lexicalGlobalObject);
        auto scope = DECLARE_CATCH_SCOPE(vm);
        JSC::JSValue returnValue;
        {
            JSExecState currentState(lexicalGlobalObject);
            returnValue = JSC::evaluate(lexicalGlobalObject, source, thisValue, returnedException);
        }
        scope.assertNoExceptionExceptTermination();
        return returnValue;
    }

    static JSC::JSValue evaluate(JSC::JSGlobalObject* lexicalGlobalObject, const JSC::SourceCode& source, JSC::JSValue thisValue = JSC::JSValue())
    {
        NakedPtr<JSC::Exception> unused;
        return evaluate(lexicalGlobalObject, source, thisValue, unused);
    }

    static JSC::JSValue profiledCall(JSC::JSGlobalObject* lexicalGlobalObject, JSC::ProfilingReason reason, JSC::JSValue functionObject, const JSC::CallData& callData, JSC::JSValue thisValue, const JSC::ArgList& args, NakedPtr<JSC::Exception>& returnedException)
    {
        JSC::VM& vm = JSC::getVM(lexicalGlobalObject);
        auto scope = DECLARE_CATCH_SCOPE(vm);
        JSC::JSValue returnValue;
        {
            JSExecState currentState(lexicalGlobalObject);
            returnValue = JSC::profiledCall(lexicalGlobalObject, reason, functionObject, callData, thisValue, args, returnedException);
        }
        scope.assertNoExceptionExceptTermination();
        return returnValue;
    }

    static JSC::JSValue profiledEvaluate(JSC::JSGlobalObject* lexicalGlobalObject, JSC::ProfilingReason reason, const JSC::SourceCode& source, JSC::JSValue thisValue, NakedPtr<JSC::Exception>& returnedException)
    {
        JSC::VM& vm = JSC::getVM(lexicalGlobalObject);
        auto scope = DECLARE_CATCH_SCOPE(vm);
        JSC::JSValue returnValue;
        {
            JSExecState currentState(lexicalGlobalObject);
            returnValue = JSC::profiledEvaluate(lexicalGlobalObject, reason, source, thisValue, returnedException);
        }
        scope.assertNoExceptionExceptTermination();
        return returnValue;
    }

    static JSC::JSValue profiledEvaluate(JSC::JSGlobalObject* lexicalGlobalObject, JSC::ProfilingReason reason, const JSC::SourceCode& source, JSC::JSValue thisValue = JSC::JSValue())
    {
        NakedPtr<JSC::Exception> unused;
        return profiledEvaluate(lexicalGlobalObject, reason, source, thisValue, unused);
    }

    static void runTask(JSC::JSGlobalObject*, JSC::QueuedTask&);

    static JSC::JSInternalPromise* loadModule(JSC::JSGlobalObject& lexicalGlobalObject, const URL& topLevelModuleURL, JSC::JSValue parameters, JSC::JSValue scriptFetcher)
    {
        JSExecState currentState(&lexicalGlobalObject);
        return JSC::loadModule(&lexicalGlobalObject, JSC::Identifier::fromString(lexicalGlobalObject.vm(), topLevelModuleURL.string()), parameters, scriptFetcher);
    }

    static JSC::JSInternalPromise* loadModule(JSC::JSGlobalObject& lexicalGlobalObject, const JSC::SourceCode& sourceCode, JSC::JSValue scriptFetcher)
    {
        JSExecState currentState(&lexicalGlobalObject);
        return JSC::loadModule(&lexicalGlobalObject, sourceCode, scriptFetcher);
    }

    static JSC::JSValue linkAndEvaluateModule(JSC::JSGlobalObject& lexicalGlobalObject, const JSC::Identifier& moduleKey, JSC::JSValue scriptFetcher, NakedPtr<JSC::Exception>& returnedException)
    {
        JSC::VM& vm = JSC::getVM(&lexicalGlobalObject);
        auto scope = DECLARE_CATCH_SCOPE(vm);
        JSC::JSValue returnValue;
        {
            JSExecState currentState(&lexicalGlobalObject);
            returnValue = JSC::linkAndEvaluateModule(&lexicalGlobalObject, moduleKey, scriptFetcher);
            if (UNLIKELY(scope.exception())) {
                returnedException = scope.exception();
                if (!vm.hasPendingTerminationException())
                    scope.clearException();
                return JSC::jsUndefined();
            }
        }
        scope.assertNoExceptionExceptTermination();
        return returnValue;
    }

    static void instrumentFunction(ScriptExecutionContext*, const JSC::CallData&);

private:
    explicit JSExecState(JSC::JSGlobalObject* lexicalGlobalObject)
        : m_previousState(currentState())
        , m_lock(lexicalGlobalObject)
    {
        setCurrentState(lexicalGlobalObject);
    };

    ~JSExecState()
    {
        JSC::VM& vm = currentState()->vm();
        auto scope = DECLARE_THROW_SCOPE(vm);
        scope.assertNoExceptionExceptTermination();

        JSC::JSGlobalObject* lexicalGlobalObject = currentState();
        bool didExitJavaScript = lexicalGlobalObject && !m_previousState;

        setCurrentState(m_previousState);

        if (didExitJavaScript) {
            didLeaveScriptContext(lexicalGlobalObject);
            // We need to clear any exceptions from microtask drain.
            if (!vm.hasPendingTerminationException())
                scope.clearException();
        }
    }

    static void setCurrentState(JSC::JSGlobalObject* lexicalGlobalObject)
    {
        threadGlobalData().setCurrentState(lexicalGlobalObject);
    }

    JSC::JSGlobalObject* const m_previousState;
    JSC::JSLockHolder m_lock;

    static void didLeaveScriptContext(JSC::JSGlobalObject*);
};

// Null lexicalGlobalObject prevents origin security checks.
// Used by non-JavaScript bindings (ObjC, GObject).
class JSMainThreadNullState {
    WTF_MAKE_NONCOPYABLE(JSMainThreadNullState);
    WTF_FORBID_HEAP_ALLOCATION;
public:
    explicit JSMainThreadNullState()
        : m_previousState(JSExecState::currentState())
        , m_customElementReactionStack(m_previousState)
    {
        ASSERT(isMainThread());
        JSExecState::setCurrentState(nullptr);
    }

    ~JSMainThreadNullState()
    {
        ASSERT(isMainThread());
        JSExecState::setCurrentState(m_previousState);
    }

private:
    JSC::JSGlobalObject* const m_previousState;
    CustomElementReactionStack m_customElementReactionStack;
};

JSC::JSValue functionCallHandlerFromAnyThread(JSC::JSGlobalObject*, JSC::JSValue functionObject, const JSC::CallData&, JSC::JSValue thisValue, const JSC::ArgList& args, NakedPtr<JSC::Exception>& returnedException);
JSC::JSValue evaluateHandlerFromAnyThread(JSC::JSGlobalObject*, const JSC::SourceCode&, JSC::JSValue thisValue, NakedPtr<JSC::Exception>& returnedException);

ScriptExecutionContext* executionContext(JSC::JSGlobalObject*);

} // namespace WebCore
