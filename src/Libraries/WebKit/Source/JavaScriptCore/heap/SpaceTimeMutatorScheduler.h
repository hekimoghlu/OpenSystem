/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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

#include "MutatorScheduler.h"
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>


namespace JSC {

class Heap;

// The JSC concurrent GC sometimes stops the world in order to stay ahead of it. These deliberate,
// synthetic pauses ensure that the GC won't have to do one huge pause in order to catch up to the
// retreating wavefront. The scheduler is called "space-time" because it links the amount of time
// that the world is paused for to the amount of space that the world allocated since the GC cycle
// began.

class SpaceTimeMutatorScheduler final : public MutatorScheduler {
    WTF_MAKE_TZONE_ALLOCATED(SpaceTimeMutatorScheduler);
public:
    SpaceTimeMutatorScheduler(Heap&);
    ~SpaceTimeMutatorScheduler() final;
    
    State state() const final;
    
    void beginCollection() final;
    
    void didStop() final;
    void willResume() final;
    void didExecuteConstraints() final;
    
    MonotonicTime timeToStop() final;
    MonotonicTime timeToResume() final;
    
    void log() final;
    
    void endCollection() final;
    
private:
    class Snapshot;
    friend class Snapshot;
    
    double bytesAllocatedThisCycleImpl();
    
    double bytesSinceBeginningOfCycle(const Snapshot&);
    double maxHeadroom();
    double headroomFullness(const Snapshot&);
    double mutatorUtilization(const Snapshot&);
    double collectorUtilization(const Snapshot&);
    Seconds elapsedInPeriod(const Snapshot&);
    double phase(const Snapshot&);
    bool shouldBeResumed(const Snapshot&);
    
    JSC::Heap& m_heap;
    Seconds m_period;
    State m_state { Normal };
    
    double m_bytesAllocatedThisCycleAtTheBeginning { 0 };
    double m_bytesAllocatedThisCycleAtTheEnd { 0 };
    MonotonicTime m_startTime;
};

} // namespace JSC

