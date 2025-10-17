/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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

#include <wtf/Compiler.h>

#include "FetchOptions.h"
#include "WorkerThreadType.h"
#include <JavaScriptCore/Debugger.h>
#include <JavaScriptCore/JSRunLoopTimer.h>
#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/Lock.h>
#include <wtf/MessageQueue.h>
#include <wtf/NakedPtr.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {
class AbstractModuleRecord;
class Exception;
class JSGlobalObject;
class JSModuleRecord;
class VM;
}

namespace WebCore {

class Exception;
class JSDOMGlobalObject;
class ScriptSourceCode;
class WorkerConsoleClient;
class WorkerOrWorkletGlobalScope;
class WorkerScriptFetcher;

class WorkerOrWorkletScriptController final : public CanMakeCheckedPtr<WorkerOrWorkletScriptController> {
    WTF_MAKE_TZONE_ALLOCATED(WorkerOrWorkletScriptController);
    WTF_MAKE_NONCOPYABLE(WorkerOrWorkletScriptController);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WorkerOrWorkletScriptController);
public:
    WorkerOrWorkletScriptController(WorkerThreadType, Ref<JSC::VM>&&, WorkerOrWorkletGlobalScope*);
    ~WorkerOrWorkletScriptController();

    void releaseHeapAccess();
    void acquireHeapAccess();

    void addTimerSetNotification(JSC::JSRunLoopTimer::TimerNotificationCallback);
    void removeTimerSetNotification(JSC::JSRunLoopTimer::TimerNotificationCallback);

    JSDOMGlobalObject* globalScopeWrapper()
    {
        initScriptIfNeeded();
        return m_globalScopeWrapper.get();
    }

    void attachDebugger(JSC::Debugger*);
    void detachDebugger(JSC::Debugger*);

    // Async request to terminate a JS run execution. Eventually causes termination
    // exception raised during JS execution, if the worker thread happens to run JS.
    // After JS execution was terminated in this way, the Worker thread has to use
    // forbidExecution()/isExecutionForbidden() to guard against reentry into JS.
    // Can be called from any thread.
    void scheduleExecutionTermination();
    bool isTerminatingExecution() const;

    // Called on Worker thread when JS exits with termination exception caused by forbidExecution() request,
    // or by Worker thread termination code to prevent future entry into JS.
    void forbidExecution();
    bool isExecutionForbidden() const;

    JSC::VM& vm() { return *m_vm; }

    void setException(JSC::Exception*);

    void disableEval(const String& errorMessage);
    void disableWebAssembly(const String& errorMessage);
    void setRequiresTrustedTypes(bool required);

    void evaluate(const ScriptSourceCode&, String* returnedExceptionMessage = nullptr);
    void evaluate(const ScriptSourceCode&, NakedPtr<JSC::Exception>& returnedException, String* returnedExceptionMessage = nullptr);

    JSC::JSValue evaluateModule(JSC::AbstractModuleRecord&, JSC::JSValue awaitedValue, JSC::JSValue resumeMode);

    void linkAndEvaluateModule(WorkerScriptFetcher&, const ScriptSourceCode&, String* returnedExceptionMessage = nullptr);
    bool loadModuleSynchronously(WorkerScriptFetcher&, const ScriptSourceCode&);

    void loadAndEvaluateModule(const URL& moduleURL, FetchOptions::Credentials, CompletionHandler<void(std::optional<Exception>&&)>&&);

protected:
    WorkerOrWorkletGlobalScope* globalScope() const { return m_globalScope; }

    void initScriptIfNeeded()
    {
        if (!m_globalScopeWrapper)
            initScript();
    }
    WEBCORE_EXPORT void initScript();

private:
    template<typename JSGlobalScopePrototype, typename JSGlobalScope, typename GlobalScope>
    void initScriptWithSubclass();

    RefPtr<JSC::VM> m_vm;
    WorkerOrWorkletGlobalScope* m_globalScope;
    JSC::Strong<JSDOMGlobalObject> m_globalScopeWrapper;
    std::unique_ptr<WorkerConsoleClient> m_consoleClient;
    mutable Lock m_scheduledTerminationLock;
    bool m_isTerminatingExecution WTF_GUARDED_BY_LOCK(m_scheduledTerminationLock) { false };
};

} // namespace WebCore
