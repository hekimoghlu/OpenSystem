/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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

#if ENABLE(WEBASSEMBLY)

#include <queue>

#include <wtf/AutomaticThread.h>
#include <wtf/PrintStream.h>
#include <wtf/PriorityQueue.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace JSC {

class VM;

namespace Wasm {

class Plan;

class Worklist {
    WTF_MAKE_TZONE_ALLOCATED(Worklist);
public:
    Worklist();
    ~Worklist();

    JS_EXPORT_PRIVATE void enqueue(Ref<Plan>);
    void stopAllPlansForContext(VM&);

    JS_EXPORT_PRIVATE void completePlanSynchronously(Plan&);

    enum class Priority {
        Shutdown,
        Synchronous,
        Compilation,
        Preparation
    };

    void dump(PrintStream&) const;

private:
    class Thread;
    friend class Thread;

    typedef uint64_t Ticket;
    Ticket nextTicket() { return m_lastGrantedTicket++; }

    struct QueueElement {
        Priority priority;
        Ticket ticket;
        RefPtr<Plan> plan;

        void setToNextPriority();
    };

    static bool isHigherPriority(const QueueElement& left, const QueueElement& right)
    {
        if (left.priority == right.priority)
            return left.ticket > right.ticket;
        return left.priority > right.priority;
    }

    Box<Lock> m_lock;
    Ref<AutomaticThreadCondition> m_planEnqueued;
    // Technically, this could overflow but that's unlikely. Even if it did, we will just compile things of the same
    // Priority it the wrong order, which isn't wrong, just suboptimal.
    Ticket m_lastGrantedTicket { 0 };
    PriorityQueue<QueueElement, isHigherPriority, 10> m_queue;
    Vector<Ref<Thread>> m_threads;
};

Worklist* existingWorklistOrNull();
JS_EXPORT_PRIVATE Worklist& ensureWorklist();

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
