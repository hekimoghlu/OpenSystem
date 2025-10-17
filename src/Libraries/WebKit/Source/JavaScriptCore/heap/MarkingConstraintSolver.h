/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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

#include "VisitCounter.h"
#include <wtf/BitVector.h>
#include <wtf/Condition.h>
#include <wtf/Deque.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/ScopedLambda.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace JSC {

class Heap;
class MarkingConstraint;
class MarkingConstraintSet;

class MarkingConstraintSolver {
    WTF_MAKE_NONCOPYABLE(MarkingConstraintSolver);
    WTF_MAKE_TZONE_ALLOCATED(MarkingConstraintSolver);
    
public:
    MarkingConstraintSolver(MarkingConstraintSet&);
    ~MarkingConstraintSolver();
    
    bool didVisitSomething() const;
    
    enum SchedulerPreference {
        ParallelWorkFirst,
        NextConstraintFirst
    };

    void execute(SchedulerPreference, ScopedLambda<std::optional<unsigned>()> pickNext);
    
    void drain(BitVector& unexecuted);
    
    void converge(const Vector<MarkingConstraint*>& order);
    
    void execute(MarkingConstraint&);
    
    // Parallel constraints can add parallel tasks.
    void addParallelTask(RefPtr<SharedTask<void(SlotVisitor&)>>, MarkingConstraint&);
    
private:
    void runExecutionThread(SlotVisitor&, SchedulerPreference, ScopedLambda<std::optional<unsigned>()> pickNext);
    
    struct TaskWithConstraint {
        TaskWithConstraint() { }
        
        TaskWithConstraint(RefPtr<SharedTask<void(SlotVisitor&)>> task, MarkingConstraint* constraint)
            : task(WTFMove(task))
            , constraint(constraint)
        {
        }
        
        friend bool operator==(const TaskWithConstraint&, const TaskWithConstraint&) = default;
        
        RefPtr<SharedTask<void(SlotVisitor&)>> task;
        MarkingConstraint* constraint { nullptr };
    };
    
    JSC::Heap& m_heap;
    SlotVisitor& m_mainVisitor;
    MarkingConstraintSet& m_set;
    BitVector m_executed;
    Deque<TaskWithConstraint, 32> m_toExecuteInParallel;
    Vector<unsigned, 32> m_toExecuteSequentially;
    Lock m_lock;
    Condition m_condition;
    bool m_pickNextIsStillActive { true };
    unsigned m_numThreadsThatMayProduceWork { 0 };
    Vector<VisitCounter, 16> m_visitCounters;
};

} // namespace JSC

