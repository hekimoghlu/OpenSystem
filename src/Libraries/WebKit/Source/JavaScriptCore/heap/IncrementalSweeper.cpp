/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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
#include "IncrementalSweeper.h"

#include "DeferGCInlines.h"
#include "HeapInlines.h"
#include "MarkedBlockInlines.h"
#include <wtf/SystemTracing.h>

namespace JSC {

static constexpr Seconds sweepTimeSlice = 10_ms;
static constexpr double sweepTimeTotal = .10;
static constexpr double sweepTimeMultiplier = 1.0 / sweepTimeTotal;

void IncrementalSweeper::scheduleTimer()
{
    setTimeUntilFire(sweepTimeSlice * sweepTimeMultiplier);
}

IncrementalSweeper::IncrementalSweeper(JSC::Heap* heap)
    : Base(heap->vm())
    , m_currentDirectory(nullptr)
{
}

void IncrementalSweeper::doWorkUntil(VM& vm, MonotonicTime deadline)
{
    if (!m_currentDirectory)
        m_currentDirectory = vm.heap.objectSpace().firstDirectory();

    if (m_currentDirectory)
        doSweep(vm, deadline, SweepTrigger::OpportunisticTask);
}

void IncrementalSweeper::doWork(VM& vm)
{
    if (m_lastOpportunisticTaskDidFinishSweeping) {
        m_lastOpportunisticTaskDidFinishSweeping = false;
        scheduleTimer();
        return;
    }
    doSweep(vm, MonotonicTime::now() + sweepTimeSlice, SweepTrigger::Timer);
}

void IncrementalSweeper::doSweep(VM& vm, MonotonicTime deadline, SweepTrigger trigger)
{
    std::optional<TraceScope> traceScope;
    if (UNLIKELY(Options::useTracePoints()))
        traceScope.emplace(IncrementalSweepStart, IncrementalSweepEnd, vm.heap.size(), vm.heap.capacity());

    while (sweepNextBlock(vm, trigger)) {
        if (MonotonicTime::now() < deadline)
            continue;

        if (trigger == SweepTrigger::Timer)
            scheduleTimer();
        else
            m_lastOpportunisticTaskDidFinishSweeping = false;
        return;
    }
    if (trigger == SweepTrigger::OpportunisticTask)
        m_lastOpportunisticTaskDidFinishSweeping = true;

    cancelTimer();
}

bool IncrementalSweeper::sweepNextBlock(VM& vm, SweepTrigger trigger)
{
    vm.heap.stopIfNecessary();

    MarkedBlock::Handle* block = nullptr;
    
    for (; m_currentDirectory; m_currentDirectory = m_currentDirectory->nextDirectory()) {
        block = m_currentDirectory->findBlockToSweep();
        if (block)
            break;
    }
    
    if (block) {
        DeferGCForAWhile deferGC(vm);
        block->sweep(nullptr);

        bool blockIsFreed = false;
        if (trigger == SweepTrigger::Timer) {
            if (!block->isEmpty())
                block->shrink();
            else {
                vm.heap.objectSpace().freeBlock(block);
                blockIsFreed = true;
            }
        }

        if (!blockIsFreed)
            m_currentDirectory->didFinishUsingBlock(block);
        return true;
    }

    return vm.heap.sweepNextLogicallyEmptyWeakBlock();
}

void IncrementalSweeper::startSweeping(JSC::Heap& heap)
{
    scheduleTimer();
    m_currentDirectory = heap.objectSpace().firstDirectory();
}

void IncrementalSweeper::stopSweeping()
{
    m_currentDirectory = nullptr;
    cancelTimer();
}

} // namespace JSC
