/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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

#if USE(COORDINATED_GRAPHICS)

#include <wtf/Atomics.h>
#include <wtf/Condition.h>
#include <wtf/Function.h>
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/Noncopyable.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {
class CompositingRunLoop;
}

namespace WTF {
template<typename T> struct IsDeprecatedTimerSmartPointerException;
template<> struct IsDeprecatedTimerSmartPointerException<WebKit::CompositingRunLoop> : std::true_type { };
}

namespace WebKit {

class CompositingRunLoop {
    WTF_MAKE_NONCOPYABLE(CompositingRunLoop);
    WTF_MAKE_TZONE_ALLOCATED(CompositingRunLoop);
public:
    CompositingRunLoop(Function<void ()>&&);
    ~CompositingRunLoop();

    bool isCurrent() const;
    bool isActive();

    void performTask(Function<void ()>&&);
    void performTaskSync(Function<void ()>&&);

    void suspend();
    void resume();

    Lock& stateLock() { return m_state.lock; }

    void scheduleUpdate();
    void stopUpdates();

    void updateCompleted(Locker<Lock>&);

    RunLoop& runLoop() const { return m_runLoop.get(); }

private:
    enum class UpdateState {
        Idle,
        Scheduled,
        InProgress,
    };

    void scheduleUpdate(Locker<Lock>&);
    void updateTimerFired();

    Ref<RunLoop> m_runLoop;
    RunLoop::Timer m_updateTimer;
    Function<void ()> m_updateFunction;
    Lock m_dispatchSyncConditionLock;
    Condition m_dispatchSyncCondition;

    struct {
        Lock lock;
        UpdateState update { UpdateState::Idle };
        bool pendingUpdate { false };
        bool isSuspended { false };
    } m_state;
};

} // namespace WebKit

#endif // USE(COORDINATED_GRAPHICS)
