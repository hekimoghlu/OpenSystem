/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 30, 2022.
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

#include "ScriptExecutionContext.h"
#include <memory>
#include <wtf/MessageQueue.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class WorkerMainRunLoop;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::WorkerMainRunLoop> : std::true_type { };
}

namespace WebCore {

class WeakPtrImplWithEventTargetData;
class ModePredicate;
class WorkerOrWorkletGlobalScope;
class WorkerSharedTimer;

class WorkerRunLoop {
    WTF_MAKE_TZONE_ALLOCATED(WorkerRunLoop);
public:
    enum class Type : bool { WorkerDedicatedRunLoop, WorkerMainRunLoop };

    virtual ~WorkerRunLoop() = default;

    virtual bool runInMode(WorkerOrWorkletGlobalScope*, const String& mode, bool allowEventLoopTasks = false) = 0;
    virtual void postTaskAndTerminate(ScriptExecutionContext::Task&&) = 0;
    virtual void postTaskForMode(ScriptExecutionContext::Task&&, const String& mode) = 0;
    virtual void terminate() = 0;
    virtual bool terminated() const = 0;
    virtual Type type() const = 0;

    void postTask(ScriptExecutionContext::Task&&);
    void postDebuggerTask(ScriptExecutionContext::Task&&);

    WEBCORE_EXPORT static String defaultMode();

    unsigned long createUniqueId() { return ++m_uniqueId; }

private:
    unsigned long m_uniqueId { 0 };
};

class WorkerDedicatedRunLoop final : public WorkerRunLoop {
    WTF_MAKE_TZONE_ALLOCATED(WorkerDedicatedRunLoop);
public:
    WorkerDedicatedRunLoop();
    ~WorkerDedicatedRunLoop();
    
    // Blocking call. Waits for tasks and timers, invokes the callbacks.
    void run(WorkerOrWorkletGlobalScope*);

    // Waits for a single task and returns.
    bool runInMode(WorkerOrWorkletGlobalScope*, const String& mode, bool) final;
    MessageQueueWaitResult runInDebuggerMode(WorkerOrWorkletGlobalScope&);

    void terminate() final;
    bool terminated() const final { return m_messageQueue.killed(); }
    Type type() const final { return Type::WorkerDedicatedRunLoop; }

    void postTaskAndTerminate(ScriptExecutionContext::Task&&) final;
    WEBCORE_EXPORT void postTaskForMode(ScriptExecutionContext::Task&&, const String& mode) final;

    class Task {
        WTF_MAKE_TZONE_ALLOCATED(Task);
        WTF_MAKE_NONCOPYABLE(Task);
    public:
        Task(ScriptExecutionContext::Task&&, const String& mode);
        const String& mode() const { return m_mode; }

    private:
        void performTask(WorkerOrWorkletGlobalScope*);

        ScriptExecutionContext::Task m_task;
        String m_mode;

        friend class WorkerDedicatedRunLoop;
    };

private:
    friend class RunLoopSetup;
    MessageQueueWaitResult runInMode(WorkerOrWorkletGlobalScope*, const ModePredicate&);

    // Runs any clean up tasks that are currently in the queue and returns.
    // This should only be called when the context is closed or loop has been terminated.
    void runCleanupTasks(WorkerOrWorkletGlobalScope*);

    bool isBeingDebugged() const { return m_debugCount >= 1; }

    MessageQueue<Task> m_messageQueue;
    std::unique_ptr<WorkerSharedTimer> m_sharedTimer;
    int m_nestedCount { 0 };
    int m_debugCount { 0 };
};

class WorkerMainRunLoop final : public WorkerRunLoop, public CanMakeWeakPtr<WorkerMainRunLoop, WeakPtrFactoryInitialization::Eager> {
public:
    WorkerMainRunLoop();

    void setGlobalScope(WorkerOrWorkletGlobalScope&);

    void terminate() final { m_terminated = true; }
    bool terminated() const final { return m_terminated; }

    bool runInMode(WorkerOrWorkletGlobalScope*, const String& mode, bool);
    void postTaskAndTerminate(ScriptExecutionContext::Task&&) final;
    void postTaskForMode(ScriptExecutionContext::Task&&, const String& mode) final;
    Type type() const final { return Type::WorkerMainRunLoop; }

private:
    WeakPtr<WorkerOrWorkletGlobalScope> m_workerOrWorkletGlobalScope;
    bool m_terminated { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::WorkerDedicatedRunLoop)
    static bool isType(const WebCore::WorkerRunLoop& runLoop) { return runLoop.type() == WebCore::WorkerRunLoop::Type::WorkerDedicatedRunLoop; }
SPECIALIZE_TYPE_TRAITS_END()

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::WorkerMainRunLoop)
    static bool isType(const WebCore::WorkerRunLoop& runLoop) { return runLoop.type() == WebCore::WorkerRunLoop::Type::WorkerMainRunLoop; }
SPECIALIZE_TYPE_TRAITS_END()
