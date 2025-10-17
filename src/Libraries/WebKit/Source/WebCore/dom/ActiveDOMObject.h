/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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

#include "ContextDestructionObserver.h"
#include "TaskSource.h"
#include <wtf/AbstractRefCounted.h>
#include <wtf/Assertions.h>
#include <wtf/CancellableTask.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/RefCounted.h>
#include <wtf/Threading.h>

namespace WebCore {

class Document;
class Event;
class EventLoopTaskGroup;
class EventTarget;

enum class ReasonForSuspension : uint8_t {
    JavaScriptDebuggerPaused,
    WillDeferLoading,
    BackForwardCache,
    PageWillBeSuspended,
};

class WEBCORE_EXPORT ActiveDOMObject : public AbstractRefCounted, public ContextDestructionObserver {
public:
    // The suspendIfNeeded must be called exactly once after object construction to update
    // the suspended state to match that of the ScriptExecutionContext.
    void suspendIfNeeded();
    void assertSuspendIfNeededWasCalled() const;

    void didMoveToNewDocument(Document&);

    // This function is used by JS bindings to determine if the JS wrapper should be kept alive or not.
    bool hasPendingActivity() const { return m_pendingActivityInstanceCount || virtualHasPendingActivity(); }

    // However, the suspend function will sometimes be called even if canSuspendForDocumentSuspension() returns false.
    // That happens in step-by-step JS debugging for example - in this case it would be incorrect
    // to stop the object. Exact semantics of suspend is up to the object in cases like that.

    // These functions must not have a side effect of creating or destroying
    // any ActiveDOMObject. That means they must not result in calls to arbitrary JavaScript.
    virtual void suspend(ReasonForSuspension);
    virtual void resume();

    // This function must not have a side effect of creating an ActiveDOMObject.
    // That means it must not result in calls to arbitrary JavaScript.
    // It can, however, have a side effect of deleting an ActiveDOMObject.
    virtual void stop();

    template<class T>
    class PendingActivity : public RefCounted<PendingActivity<T>> {
    public:
        explicit PendingActivity(T& thisObject)
            : m_thisObject(thisObject)
        {
            ++(m_thisObject->m_pendingActivityInstanceCount);
        }

        ~PendingActivity()
        {
            ASSERT(m_thisObject->m_pendingActivityInstanceCount > 0);
            --(m_thisObject->m_pendingActivityInstanceCount);
        }

    private:
        Ref<T> m_thisObject;
    };

    template<class T> Ref<PendingActivity<T>> makePendingActivity(T& thisObject)
    {
        ASSERT(&thisObject == this);
        return adoptRef(*new PendingActivity<T>(thisObject));
    }

    bool isContextStopped() const;
    bool isAllowedToRunScript() const;

    template<typename T>
    static void queueTaskKeepingObjectAlive(T& object, TaskSource source, Function<void ()>&& task)
    {
        // Calls the template member function outside of lambda init-captures to work around a MSVC bug.
        auto activity = object.ActiveDOMObject::makePendingActivity(object);
        object.queueTaskInEventLoop(source, [protectedObject = Ref { object }, activity = WTFMove(activity), task = WTFMove(task)] () {
            task();
        });
    }

    template<typename T>
    static void queueCancellableTaskKeepingObjectAlive(T& object, TaskSource source, TaskCancellationGroup& cancellationGroup, Function<void()>&& task)
    {
        CancellableTask cancellableTask(cancellationGroup, WTFMove(task));
        // Calls the template member function outside of lambda init-captures to work around a MSVC bug.
        auto activity = object.ActiveDOMObject::makePendingActivity(object);
        object.queueTaskInEventLoop(source, [protectedObject = Ref { object }, activity = WTFMove(activity), cancellableTask = WTFMove(cancellableTask)]() mutable {
            cancellableTask();
        });
    }

    template<typename EventTargetType>
    static void queueTaskToDispatchEvent(EventTargetType& target, TaskSource source, Ref<Event>&& event)
    {
        target.queueTaskToDispatchEventInternal(target, source, WTFMove(event));
    }

    template<typename EventTargetType>
    static void queueCancellableTaskToDispatchEvent(EventTargetType& target, TaskSource source, TaskCancellationGroup& cancellationGroup, Ref<Event>&& event)
    {
        target.queueCancellableTaskToDispatchEventInternal(target, source, cancellationGroup, WTFMove(event));
    }

protected:
    explicit ActiveDOMObject(ScriptExecutionContext*);
    explicit ActiveDOMObject(Document*);
    explicit ActiveDOMObject(Document&);
    virtual ~ActiveDOMObject();

private:
    enum CheckedScriptExecutionContextType { CheckedScriptExecutionContext };
    ActiveDOMObject(ScriptExecutionContext*, CheckedScriptExecutionContextType);

    // This is used by subclasses to indicate that they have pending activity, meaning that they would
    // like the JS wrapper to stay alive (because they may still fire JS events).
    virtual bool virtualHasPendingActivity() const { return false; }

    void queueTaskInEventLoop(TaskSource, Function<void ()>&&);
    void queueTaskToDispatchEventInternal(EventTarget&, TaskSource, Ref<Event>&&);
    void queueCancellableTaskToDispatchEventInternal(EventTarget&, TaskSource, TaskCancellationGroup&, Ref<Event>&&);

    uint64_t m_pendingActivityInstanceCount { 0 };
#if ASSERT_ENABLED
    bool m_suspendIfNeededWasCalled { false };
    Ref<Thread> m_creationThread { Thread::current() };
#endif

    friend class ActiveDOMObjectEventDispatchTask;
};

#if !ASSERT_ENABLED

inline void ActiveDOMObject::assertSuspendIfNeededWasCalled() const
{
}

#endif

} // namespace WebCore
