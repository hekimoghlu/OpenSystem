/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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

#include "WorkerRunLoop.h"
#include "WorkerThreadMode.h"
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/FunctionDispatcher.h>
#include <wtf/Lock.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/ThreadSafeWeakHashSet.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/threads/BinarySemaphore.h>

namespace WTF {
class Thread;
}

namespace WebCore {

class WorkerDebuggerProxy;
class WorkerLoaderProxy;

class WorkerOrWorkletThread : public SerialFunctionDispatcher {
public:
    virtual ~WorkerOrWorkletThread();

    // SerialFunctionDispatcher methods
    void dispatch(Function<void()>&&) final;
    bool isCurrent() const final;

    Thread* thread() const { return m_thread.get(); }

    virtual void clearProxies() = 0;

    virtual WorkerDebuggerProxy* workerDebuggerProxy() const = 0;
    virtual WorkerLoaderProxy* workerLoaderProxy() = 0;

    WorkerOrWorkletGlobalScope* globalScope() const { return m_globalScope.get(); }
    WorkerRunLoop& runLoop() { return m_runLoop; }

    void start(Function<void(const String&)>&& evaluateCallback = { });
    void stop(Function<void()>&& terminatedCallback = { });

    void startRunningDebuggerTasks();
    void stopRunningDebuggerTasks();

    void suspend();
    void resume();

    const String& inspectorIdentifier() const { return m_inspectorIdentifier; }

    static ThreadSafeWeakHashSet<WorkerOrWorkletThread>& workerOrWorkletThreads();
    static void releaseFastMallocFreeMemoryInAllThreads();

    void addChildThread(WorkerOrWorkletThread&);
    void removeChildThread(WorkerOrWorkletThread&);

protected:
    explicit WorkerOrWorkletThread(const String& inspectorIdentifier, WorkerThreadMode = WorkerThreadMode::CreateNewThread);
    void workerOrWorkletThread();

    // Executes the event loop for the worker thread. Derived classes can override to perform actions before/after entering the event loop.
    virtual void runEventLoop();

private:
    virtual Ref<Thread> createThread() = 0;
    virtual RefPtr<WorkerOrWorkletGlobalScope> createGlobalScope() = 0;
    virtual void evaluateScriptIfNecessary(String&) { }
    virtual bool shouldWaitForWebInspectorOnStartup() const { return false; }
    void destroyWorkerGlobalScope(Ref<WorkerOrWorkletThread>&& protectedThis);

    String m_inspectorIdentifier;
    Lock m_threadCreationAndGlobalScopeLock;
    RefPtr<WorkerOrWorkletGlobalScope> m_globalScope;
    RefPtr<Thread> m_thread;
    UniqueRef<WorkerRunLoop> m_runLoop;
    Function<void(const String&)> m_evaluateCallback;
    Function<void()> m_stoppedCallback;
    BinarySemaphore m_suspensionSemaphore;
    ThreadSafeWeakHashSet<WorkerOrWorkletThread> m_childThreads;
    Function<void()> m_runWhenLastChildThreadIsGone;
    bool m_isSuspended { false };
    bool m_pausedForDebugger { false };
};

} // namespace WebCore
