/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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
#include "StaticPerProcess.h"
#include "Vector.h"
#include <chrono>
#include <condition_variable>
#include <mutex>

#if BOS(DARWIN)
#include <dispatch/dispatch.h>
#endif

#if !BUSE(LIBPAS)

namespace bmalloc {

class Scavenger : public StaticPerProcess<Scavenger> {
public:
    BEXPORT Scavenger(const LockHolder&);
    
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

    // Used for debugging only.
    void disable() { m_isEnabled = false; }

private:
    enum class State { Sleep, Run, RunSoon };
    
    void run(const LockHolder&);
    void runSoon(const LockHolder&);

    void scheduleIfUnderMemoryPressure(const LockHolder&, size_t bytes);

    BNO_RETURN static void threadEntryPoint(Scavenger*);
    BNO_RETURN void threadRunLoop();
    
    void setSelfQOSClass();
    void setThreadName(const char*);

    std::chrono::milliseconds timeSinceLastFullScavenge();

    std::atomic<State> m_state { State::Sleep };
    size_t m_scavengerBytes { 0 };
    std::chrono::milliseconds m_waitTime;
    bool m_isInMiniMode { false };
    
    Mutex m_scavengingMutex;
    std::condition_variable_any m_condition;

    std::thread m_thread;
    std::chrono::steady_clock::time_point m_lastFullScavengeTime { std::chrono::steady_clock::now() };

#if BOS(DARWIN)
    dispatch_source_t m_pressureHandlerDispatchSource;
    qos_class_t m_requestedScavengerThreadQOSClass { QOS_CLASS_USER_INITIATED };
#endif

#if BPLATFORM(MAC)
    const unsigned s_newWaitMultiplier = 300;
    const unsigned s_minWaitTimeMilliseconds = 750;
    const unsigned s_maxWaitTimeMilliseconds = 20000;
#else
    const unsigned s_newWaitMultiplier = 150;
    const unsigned s_minWaitTimeMilliseconds = 100;
    const unsigned s_maxWaitTimeMilliseconds = 10000;
#endif

    Vector<DeferredDecommit> m_deferredDecommits;
    bool m_isEnabled { true };
};
BALLOW_DEPRECATED_DECLARATIONS_BEGIN
DECLARE_STATIC_PER_PROCESS_STORAGE(Scavenger);
BALLOW_DEPRECATED_DECLARATIONS_END

} // namespace bmalloc

#endif
