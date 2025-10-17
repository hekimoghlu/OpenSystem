/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 10, 2024.
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
#include "MarkingConstraintSolver.h"

#include "JSCInlines.h"
#include "MarkingConstraintSet.h"

namespace JSC { 

MarkingConstraintSolver::MarkingConstraintSolver(MarkingConstraintSet& set)
    : m_heap(set.m_heap)
    , m_mainVisitor(m_heap.collectorSlotVisitor())
    , m_set(set)
{
    m_heap.forEachSlotVisitor(
        [&] (SlotVisitor& visitor) {
            m_visitCounters.append(VisitCounter(visitor));
        });
}

MarkingConstraintSolver::~MarkingConstraintSolver() = default;

bool MarkingConstraintSolver::didVisitSomething() const
{
    for (const VisitCounter& visitCounter : m_visitCounters) {
        if (visitCounter.visitCount())
            return true;
    }
    return false;
}

void MarkingConstraintSolver::execute(SchedulerPreference preference, ScopedLambda<std::optional<unsigned>()> pickNext)
{
    m_pickNextIsStillActive = true;
    RELEASE_ASSERT(!m_numThreadsThatMayProduceWork);
    
    if (Options::useParallelMarkingConstraintSolver()) {
        dataLogIf(Options::logGC(), preference == ParallelWorkFirst ? "P" : "N", "<");
        
        m_heap.runFunctionInParallel(
            [&] (SlotVisitor& visitor) { runExecutionThread(visitor, preference, pickNext); });
        
        dataLogIf(Options::logGC(), ">");
    } else
        runExecutionThread(m_mainVisitor, preference, pickNext);
    
    RELEASE_ASSERT(!m_pickNextIsStillActive);
    RELEASE_ASSERT(!m_numThreadsThatMayProduceWork);
        
    if (!m_toExecuteSequentially.isEmpty()) {
        for (unsigned indexToRun : m_toExecuteSequentially)
            execute(*m_set.m_set[indexToRun]);
        m_toExecuteSequentially.clear();
    }
        
    RELEASE_ASSERT(m_toExecuteInParallel.isEmpty());
}

void MarkingConstraintSolver::drain(BitVector& unexecuted)
{
    auto iter = unexecuted.begin();
    auto end = unexecuted.end();
    if (iter == end)
        return;
    auto pickNext = scopedLambda<std::optional<unsigned>()>(
        [&] () -> std::optional<unsigned> {
            if (iter == end)
                return std::nullopt;
            return *iter++;
        });
    execute(NextConstraintFirst, pickNext);
    unexecuted.clearAll();
}

void MarkingConstraintSolver::converge(const Vector<MarkingConstraint*>& order)
{
    if (didVisitSomething())
        return;
    
    if (order.isEmpty())
        return;
        
    size_t index = 0;

    // We want to execute the first constraint sequentially if we think it will quickly give us a
    // result. If we ran it in parallel to other constraints, then we might end up having to wait for
    // those other constraints to finish, which would be a waste of time since during convergence it's
    // empirically most optimal to return to draining as soon as a constraint generates work. Most
    // constraints don't generate any work most of the time, and when they do generate work, they tend
    // to generate enough of it to feed a decent draining cycle. Therefore, pause times are lowest if
    // we get the heck out of here as soon as a constraint generates work. I think that part of what
    // makes this optimal is that we also never abort running a constraint early, so when we do run
    // one, it has an opportunity to generate as much work as it possibly can.
    if (order[index]->quickWorkEstimate(m_mainVisitor) > 0.) {
        execute(*order[index++]);
        
        if (m_toExecuteInParallel.isEmpty()
            && (order.isEmpty() || didVisitSomething()))
            return;
    }
    
    auto pickNext = scopedLambda<std::optional<unsigned>()>(
        [&] () -> std::optional<unsigned> {
            if (didVisitSomething())
                return std::nullopt;
            
            if (index >= order.size())
                return std::nullopt;
            
            MarkingConstraint& constraint = *order[index++];
            return constraint.index();
        });
    
    execute(ParallelWorkFirst, pickNext);
}

void MarkingConstraintSolver::execute(MarkingConstraint& constraint)
{
    if (m_executed.get(constraint.index()))
        return;
    
    constraint.prepareToExecute(NoLockingNecessary, m_mainVisitor);
    constraint.execute(m_mainVisitor);
    m_executed.set(constraint.index());
}

void MarkingConstraintSolver::addParallelTask(RefPtr<SharedTask<void(SlotVisitor&)>> task, MarkingConstraint& constraint)
{
    Locker locker { m_lock };
    m_toExecuteInParallel.append(TaskWithConstraint(WTFMove(task), &constraint));
}

void MarkingConstraintSolver::runExecutionThread(SlotVisitor& visitor, SchedulerPreference preference, ScopedLambda<std::optional<unsigned>()> pickNext)
{
    for (;;) {
        bool doParallelWorkMode;
        MarkingConstraint* constraint = nullptr;
        unsigned indexToRun = UINT_MAX;
        TaskWithConstraint task;
        {
            Locker locker { m_lock };
                        
            for (;;) {
                auto tryParallelWork = [&] () -> bool {
                    if (m_toExecuteInParallel.isEmpty())
                        return false;
                    
                    task = m_toExecuteInParallel.first();
                    constraint = task.constraint;
                    doParallelWorkMode = true;
                    return true;
                };
                
                auto tryNextConstraint = [&] () -> bool {
                    if (!m_pickNextIsStillActive)
                        return false;
                    
                    for (;;) {
                        std::optional<unsigned> pickResult = pickNext();
                        if (!pickResult) {
                            m_pickNextIsStillActive = false;
                            return false;
                        }
                        
                        if (m_executed.get(*pickResult))
                            continue;
                                    
                        MarkingConstraint& candidateConstraint = *m_set.m_set[*pickResult];
                        if (candidateConstraint.concurrency() == ConstraintConcurrency::Sequential) {
                            m_toExecuteSequentially.append(*pickResult);
                            continue;
                        }
                        if (candidateConstraint.parallelism() == ConstraintParallelism::Parallel)
                            m_numThreadsThatMayProduceWork++;
                        indexToRun = *pickResult;
                        constraint = &candidateConstraint;
                        doParallelWorkMode = false;
                        constraint->prepareToExecute(locker, visitor);
                        return true;
                    }
                };
                
                if (preference == ParallelWorkFirst) {
                    if (tryParallelWork() || tryNextConstraint())
                        break;
                } else {
                    if (tryNextConstraint() || tryParallelWork())
                        break;
                }
                
                // This means that we have nothing left to run. The only way for us to have more work is
                // if someone is running a constraint that may produce parallel work.
                
                if (!m_numThreadsThatMayProduceWork)
                    return;
                
                // FIXME: Any waiting could be replaced with just running the SlotVisitor.
                // I wonder if that would be profitable.
                m_condition.wait(m_lock);
            }
        }
                    
        if (doParallelWorkMode)
            constraint->doParallelWork(visitor, *task.task);
        else {
            if (constraint->parallelism() == ConstraintParallelism::Parallel) {
                visitor.m_currentConstraint = constraint;
                visitor.m_currentSolver = this;
            }
            
            constraint->execute(visitor);
            
            visitor.m_currentConstraint = nullptr;
            visitor.m_currentSolver = nullptr;
        }
        
        {
            Locker locker { m_lock };
            
            if (doParallelWorkMode) {
                if (!m_toExecuteInParallel.isEmpty()
                    && task == m_toExecuteInParallel.first())
                    m_toExecuteInParallel.takeFirst();
                else
                    ASSERT(!m_toExecuteInParallel.contains(task));
            } else {
                if (constraint->parallelism() == ConstraintParallelism::Parallel)
                    m_numThreadsThatMayProduceWork--;
                m_executed.set(indexToRun);
            }
                        
            m_condition.notifyAll();
        }
    }
}

} // namespace JSC

