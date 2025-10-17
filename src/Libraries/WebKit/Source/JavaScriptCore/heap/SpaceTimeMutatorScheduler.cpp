/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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
#include "SpaceTimeMutatorScheduler.h"

#include "JSCInlines.h"
#include <wtf/TZoneMallocInlines.h>


namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SpaceTimeMutatorScheduler);

// The scheduler will often make decisions based on state that is in flux. It will be fine so
// long as multiple uses of the same value all see the same value. We wouldn't get this for free,
// since our need to modularize the calculation results in a tendency to access the same mutable
// field in Heap multiple times, and to access the current time multiple times.
class SpaceTimeMutatorScheduler::Snapshot {
public:
    Snapshot(SpaceTimeMutatorScheduler& scheduler)
    {
        m_now = MonotonicTime::now();
        m_bytesAllocatedThisCycle = scheduler.bytesAllocatedThisCycleImpl();
    }
    
    MonotonicTime now() const { return m_now; }
    
    double bytesAllocatedThisCycle() const { return m_bytesAllocatedThisCycle; }
    
private:
    MonotonicTime m_now;
    double m_bytesAllocatedThisCycle;
};

SpaceTimeMutatorScheduler::SpaceTimeMutatorScheduler(JSC::Heap& heap)
    : m_heap(heap)
    , m_period(Seconds::fromMilliseconds(Options::concurrentGCPeriodMS()))
{
}

SpaceTimeMutatorScheduler::~SpaceTimeMutatorScheduler() = default;

MutatorScheduler::State SpaceTimeMutatorScheduler::state() const
{
    return m_state;
}

void SpaceTimeMutatorScheduler::beginCollection()
{
    RELEASE_ASSERT(m_state == Normal);
    m_state = Stopped;
    m_startTime = MonotonicTime::now();

    m_bytesAllocatedThisCycleAtTheBeginning = bytesAllocatedThisCycleImpl();
    m_bytesAllocatedThisCycleAtTheEnd = 
        Options::concurrentGCMaxHeadroom() *
        std::max<double>(m_bytesAllocatedThisCycleAtTheBeginning, m_heap.m_maxEdenSize);
}

void SpaceTimeMutatorScheduler::didStop()
{
    RELEASE_ASSERT(m_state == Stopped || m_state == Resumed);
    m_state = Stopped;
}

void SpaceTimeMutatorScheduler::willResume()
{
    RELEASE_ASSERT(m_state == Stopped || m_state == Resumed);
    m_state = Resumed;
}

void SpaceTimeMutatorScheduler::didExecuteConstraints()
{
    // If we execute constraints, we want to forgive the GC for all of the time it had stopped the
    // world for in this increment. This hack is empirically better than every other heuristic I
    // tried, because it just means that the GC is happy to pause for longer when it's dealing
    // with things that don't play well with concurrency.
    // FIXME: The feels so wrong but benchmarks so good.
    // https://bugs.webkit.org/show_bug.cgi?id=166833
    m_startTime = MonotonicTime::now();
}

MonotonicTime SpaceTimeMutatorScheduler::timeToStop()
{
    switch (m_state) {
    case Normal:
        return MonotonicTime::infinity();
    case Stopped:
        return MonotonicTime::now();
    case Resumed: {
        Snapshot snapshot(*this);
        if (!shouldBeResumed(snapshot))
            return snapshot.now();
        return snapshot.now() - elapsedInPeriod(snapshot) + m_period;
    } }
    
    RELEASE_ASSERT_NOT_REACHED();
    return MonotonicTime();
}

MonotonicTime SpaceTimeMutatorScheduler::timeToResume()
{
    switch (m_state) {
    case Normal:
    case Resumed:
        return MonotonicTime::now();
    case Stopped: {
        Snapshot snapshot(*this);
        if (shouldBeResumed(snapshot))
            return snapshot.now();
        return snapshot.now() - elapsedInPeriod(snapshot) + m_period * collectorUtilization(snapshot);
    } }
    
    RELEASE_ASSERT_NOT_REACHED();
    return MonotonicTime();
}

void SpaceTimeMutatorScheduler::log()
{
    ASSERT(Options::logGC());
    Snapshot snapshot(*this);
    dataLog(
        "a=", format("%.0lf", bytesSinceBeginningOfCycle(snapshot) / 1024), "kb ",
        "hf=", format("%.3lf", headroomFullness(snapshot)), " ",
        "mu=", format("%.3lf", mutatorUtilization(snapshot)), " ");
}

void SpaceTimeMutatorScheduler::endCollection()
{
    m_state = Normal;
    m_startTime = MonotonicTime::now();
}

double SpaceTimeMutatorScheduler::bytesAllocatedThisCycleImpl()
{
    return m_heap.totalBytesAllocatedThisCycle();
}

double SpaceTimeMutatorScheduler::bytesSinceBeginningOfCycle(const Snapshot& snapshot)
{
    return snapshot.bytesAllocatedThisCycle() - m_bytesAllocatedThisCycleAtTheBeginning;
}

double SpaceTimeMutatorScheduler::maxHeadroom()
{
    return m_bytesAllocatedThisCycleAtTheEnd - m_bytesAllocatedThisCycleAtTheBeginning;
}

double SpaceTimeMutatorScheduler::headroomFullness(const Snapshot& snapshot)
{
    double result = bytesSinceBeginningOfCycle(snapshot) / maxHeadroom();

    // headroomFullness can be NaN and other interesting things if
    // bytesAllocatedThisCycleAtTheBeginning is zero. We see that in debug tests. This code
    // defends against all floating point dragons.
    
    if (!(result >= 0))
        result = 0;
    if (!(result <= 1))
        result = 1;

    return result;
}

double SpaceTimeMutatorScheduler::mutatorUtilization(const Snapshot& snapshot)
{
    double mutatorUtilization = 1 - headroomFullness(snapshot);
    
    // Scale the mutator utilization into the permitted window.
    mutatorUtilization =
        Options::minimumMutatorUtilization() +
        mutatorUtilization * (
            Options::maximumMutatorUtilization() -
            Options::minimumMutatorUtilization());
    
    return mutatorUtilization;
}

double SpaceTimeMutatorScheduler::collectorUtilization(const Snapshot& snapshot)
{
    return 1 - mutatorUtilization(snapshot);
}

Seconds SpaceTimeMutatorScheduler::elapsedInPeriod(const Snapshot& snapshot)
{
    return (snapshot.now() - m_startTime) % m_period;
}

double SpaceTimeMutatorScheduler::phase(const Snapshot& snapshot)
{
    return elapsedInPeriod(snapshot) / m_period;
}

bool SpaceTimeMutatorScheduler::shouldBeResumed(const Snapshot& snapshot)
{
    return phase(snapshot) > collectorUtilization(snapshot);
}

} // namespace JSC

