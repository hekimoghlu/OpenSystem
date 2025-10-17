/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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

#include "BPlatform.h"
#include "DeferredDecommit.h"
#include "Mutex.h"
#include "PerProcess.h"
#include "Vector.h"
#include <chrono>
#include <condition_variable>
#include <mutex>

#if BOS(DARWIN)
#include <dispatch/dispatch.h>
#endif

namespace bmalloc {

class Scavenger {
public:
    BEXPORT Scavenger(std::lock_guard<Mutex>&);
    
    ~Scavenger() = delete;
    
    void scavenge();
    
#if BOS(DARWIN)
    void setScavengerThreadQOSClass(qos_class_t overrideClass) { m_requestedScavengerThreadQOSClass = overrideClass; }
    qos_class_t requestedScavengerThreadQOSClass() const { return m_requestedScavengerThreadQOSClass; }
#endif
    
    bool willRun() { return m_state == State::Run; }
    void run();
    
    bool willRunSoon() { return m_state > State::Sleep; }
    void runSoon();
    
    BEXPORT void didStartGrowing();
    BEXPORT void scheduleIfUnderMemoryPressure(size_t bytes);
    BEXPORT void schedule(size_t bytes);

    // This is only here for debugging purposes.
    // FIXME: Make this fast so we can use it to help determine when to
    // run the scavenger:
    // https://bugs.webkit.org/show_bug.cgi?id=184176
    size_t freeableMemory();
    // This doesn't do any synchronization, so it might return a slightly out of date answer.
    // It's unlikely, but possible.
    size_t footprint();

    void enableMiniMode();

private:
    enum class State { Sleep, Run, RunSoon };
    
    void runHoldingLock();
    void runSoonHoldingLock();

    void scheduleIfUnderMemoryPressureHoldingLock(size_t bytes);

    BNO_RETURN static void threadEntryPoint(Scavenger*);
    BNO_RETURN void threadRunLoop();
    
    void setSelfQOSClass();
    void setThreadName(const char*);

    std::chrono::milliseconds timeSinceLastFullScavenge();
    std::chrono::milliseconds timeSinceLastPartialScavenge();
    void partialScavenge();

    std::atomic<State> m_state { State::Sleep };
    size_t m_scavengerBytes { 0 };
    bool m_isProbablyGrowing { false };
    
    Mutex m_mutex;
    Mutex m_scavengingMutex;
    std::condition_variable_any m_condition;

    std::thread m_thread;
    std::chrono::steady_clock::time_point m_lastFullScavengeTime { std::chrono::steady_clock::now() };
    std::chrono::steady_clock::time_point m_lastPartialScavengeTime { std::chrono::steady_clock::now() };
    
#if BOS(DARWIN)
    dispatch_source_t m_pressureHandlerDispatchSource;
    qos_class_t m_requestedScavengerThreadQOSClass { QOS_CLASS_USER_INITIATED };
#endif
    
    Vector<DeferredDecommit> m_deferredDecommits;

    bool m_isInMiniMode { false };
};

} // namespace bmalloc


