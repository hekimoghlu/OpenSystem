/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#include <wtf/WeakRandom.h>

namespace JSC {

class Heap;

// The JSC concurrent GC sometimes stops the world in order to stay ahead of it. These deliberate,
// synthetic pauses ensure that the GC won't have to do one huge pause in order to catch up to the
// retreating wavefront. The scheduler is called "space-time" because it links the amount of time
// that the world is paused for to the amount of space that the world allocated since the GC cycle
// began.

class StochasticSpaceTimeMutatorScheduler final : public MutatorScheduler {
    WTF_MAKE_TZONE_ALLOCATED(StochasticSpaceTimeMutatorScheduler);
public:
    StochasticSpaceTimeMutatorScheduler(Heap&);
    ~StochasticSpaceTimeMutatorScheduler() final;
    
    State state() const final;
    
    void beginCollection() final;
    
    void didStop() final;
    void willResume() final;
    void didReachTermination() final;
    void didExecuteConstraints() final;
    void synchronousDrainingDidStall() final;
    
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
    
    JSC::Heap& m_heap;
    State m_state { Normal };
    
    WeakRandom m_random;
    
    Seconds m_minimumPause;
    double m_pauseScale;
    Seconds m_targetPause;
    
    double m_bytesAllocatedThisCycleAtTheBeginning { 0 };
    double m_bytesAllocatedThisCycleAtTheEnd { 0 };
    
    MonotonicTime m_beforeConstraints;
    MonotonicTime m_plannedResumeTime;
};

} // namespace JSC

