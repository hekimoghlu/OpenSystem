/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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

#include "JSRunLoopTimer.h"
#include "WeakInlines.h"

#include <wtf/Deque.h>
#include <wtf/FixedVector.h>
#include <wtf/HashSet.h>
#include <wtf/RefPtr.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/Vector.h>

namespace JSC {

class VM;
class JSCell;
class JSObject;
class JSGlobalObject;

class DeferredWorkTimer final : public JSRunLoopTimer {
public:
    using Base = JSRunLoopTimer;

    enum class WorkType : uint8_t {
        ImminentlyScheduled,
        AtSomePoint,
    };

    class TicketData : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<TicketData>  {
    private:
        WTF_MAKE_TZONE_ALLOCATED(TicketData);
        WTF_MAKE_NONCOPYABLE(TicketData);
    public:
        inline static Ref<TicketData> create(WorkType, JSObject* scriptExecutionOwner, Vector<JSCell*>&& dependencies);

        WorkType type() const { return m_type; }
        inline VM& vm();
        JSObject* target();
        bool isTargetObject();
        inline const FixedVector<JSCell*>& dependencies(bool mayBeCancelled = false);
        inline JSObject* scriptExecutionOwner();

        inline void cancel();
        // We should not modify dependencies unless it is legitimate to do so during the end of GC.
        inline void cancelAndClear();
        bool isCancelled() const { return m_isCancelled; }

    private:
        inline TicketData(WorkType, JSObject* scriptExecutionOwner, Vector<JSCell*>&& dependencies);

        WorkType m_type;
        FixedVector<JSCell*> m_dependencies;
        JSObject* m_scriptExecutionOwner { nullptr };
        bool m_isCancelled { false };
    };

    using Ticket = TicketData*;

    void doWork(VM&) final;

    JS_EXPORT_PRIVATE Ticket addPendingWork(WorkType, VM&, JSObject* target, Vector<JSCell*>&& dependencies);
    void cancelPendingWork(VM&);

    JS_EXPORT_PRIVATE bool hasAnyPendingWork() const;
    JS_EXPORT_PRIVATE bool hasImminentlyScheduledWork() const;
    bool hasPendingWork(Ticket);
    bool hasDependencyInPendingWork(Ticket, JSCell* dependency);
    bool cancelPendingWork(Ticket);
    void cancelPendingWorkSafe(JSGlobalObject*);

    // If the script execution owner your ticket is associated with gets canceled
    // the Task will not be called and will be deallocated. So it's important
    // to make sure your memory ownership model won't leak memory when
    // this occurs. The easiest way is to make sure everything is either owned
    // by a GC'd value in dependencies or by the Task lambda.
    using Task = Function<void(Ticket)>;
    JS_EXPORT_PRIVATE void scheduleWorkSoon(Ticket, Task&&);
    JS_EXPORT_PRIVATE void didResumeScriptExecutionOwner();

    void stopRunningTasks() { m_runTasks = false; }
    JS_EXPORT_PRIVATE void runRunLoop();

    static Ref<DeferredWorkTimer> create(VM& vm) { return adoptRef(*new DeferredWorkTimer(vm)); }
private:
    DeferredWorkTimer(VM&);

    Lock m_taskLock;
    bool m_runTasks { true };
    bool m_shouldStopRunLoopWhenAllTicketsFinish { false };
    bool m_currentlyRunningTask { false };
    Deque<std::tuple<Ticket, Task>> m_tasks WTF_GUARDED_BY_LOCK(m_taskLock);
    UncheckedKeyHashSet<Ref<TicketData>> m_pendingTickets;
};

inline JSObject* DeferredWorkTimer::TicketData::target()
{
    ASSERT(!isCancelled() && isTargetObject());
    // This function can be triggered on the main thread with a GC end phase
    // and a sweeping state. So, jsCast is not wanted here.
    return std::bit_cast<JSObject*>(m_dependencies.last());
}

inline const FixedVector<JSCell*>& DeferredWorkTimer::TicketData::dependencies(bool mayBeCancelled)
{
    ASSERT_UNUSED(mayBeCancelled, mayBeCancelled || !isCancelled());
    return m_dependencies;
}

inline JSObject* DeferredWorkTimer::TicketData::scriptExecutionOwner()
{
    ASSERT(!isCancelled());
    return m_scriptExecutionOwner;
}

} // namespace JSC
