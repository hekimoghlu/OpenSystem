/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 27, 2022.
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
#include "CompositingRunLoop.h"

#if USE(COORDINATED_GRAPHICS)

#include <wtf/HashMap.h>
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Threading.h>
#include <wtf/threads/BinarySemaphore.h>

#if USE(GLIB_EVENT_LOOP)
#include <wtf/glib/RunLoopSourcePriority.h>
#endif

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CompositingRunLoop);

CompositingRunLoop::CompositingRunLoop(Function<void ()>&& updateFunction)
    : m_runLoop(RunLoop::create("org.webkit.ThreadedCompositor"_s, ThreadType::Graphics))
    , m_updateTimer(m_runLoop.get(), this, &CompositingRunLoop::updateTimerFired)
    , m_updateFunction(WTFMove(updateFunction))
{
#if USE(GLIB_EVENT_LOOP)
    m_updateTimer.setPriority(RunLoopSourcePriority::CompositingThreadUpdateTimer);
    m_updateTimer.setName("[WebKit] CompositingRunLoop"_s);
#endif
}

CompositingRunLoop::~CompositingRunLoop()
{
    ASSERT(RunLoop::isMain());
    // Make sure the RunLoop is stopped after the CompositingRunLoop, because m_updateTimer has a reference.
    RunLoop::main().dispatch([runLoop = m_runLoop] {
        runLoop->stop();
        runLoop->dispatch([] {
            RunLoop::current().stop();
        });
    });
}

bool CompositingRunLoop::isCurrent() const
{
    return m_runLoop->isCurrent();
}

bool CompositingRunLoop::isActive()
{
    Locker stateLocker { m_state.lock };
    return m_state.update != UpdateState::Idle;
}

void CompositingRunLoop::performTask(Function<void ()>&& function)
{
    ASSERT(RunLoop::isMain());
    m_runLoop->dispatch(WTFMove(function));
}

void CompositingRunLoop::performTaskSync(Function<void ()>&& function)
{
    ASSERT(RunLoop::isMain());
    Locker locker { m_dispatchSyncConditionLock };
    m_runLoop->dispatch([this, function = WTFMove(function)] {
        function();
        Locker locker { m_dispatchSyncConditionLock };
        m_dispatchSyncCondition.notifyOne();
    });
    m_dispatchSyncCondition.wait(m_dispatchSyncConditionLock);
}

void CompositingRunLoop::suspend()
{
    Locker stateLocker { m_state.lock };
    m_state.isSuspended = true;
    m_updateTimer.stop();
}

void CompositingRunLoop::resume()
{
    Locker stateLocker { m_state.lock };
    m_state.isSuspended = false;
    if (m_state.update == UpdateState::Scheduled)
        m_updateTimer.startOneShot(0_s);
}

void CompositingRunLoop::scheduleUpdate()
{
    Locker stateLocker { m_state.lock };
    scheduleUpdate(stateLocker);
}

void CompositingRunLoop::scheduleUpdate(Locker<Lock>& stateLocker)
{
    // An update was requested. Depending on the state:
    //  - if Idle, enter the Scheduled state and start the update timer,
    //  - if Scheduled, do nothing,
    //  - if InProgress mark an update as pending, meaning another update will be
    //    scheduled as soon as the current one is completed.

    UNUSED_PARAM(stateLocker);

    switch (m_state.update) {
    case UpdateState::Idle:
        m_state.update = UpdateState::Scheduled;
        if (!m_state.isSuspended)
            m_updateTimer.startOneShot(0_s);
        return;
    case UpdateState::Scheduled:
        return;
    case UpdateState::InProgress:
        m_state.pendingUpdate = true;
        return;
    }
}

void CompositingRunLoop::stopUpdates()
{
    // Stop everything.

    Locker locker { m_state.lock };
    m_updateTimer.stop();
    m_state.update = UpdateState::Idle;
    m_state.pendingUpdate = false;
}

void CompositingRunLoop::updateCompleted(Locker<Lock>& stateLocker)
{
    // Scene update has been signaled as completed. Depending on the state:
    //  - if Idle, Scheduled or InProgress, do nothing,
    //  - if InProgress, schedule a new update in case a pending update was marked,
    //    otherwise push the scene update state into Idle.

    UNUSED_PARAM(stateLocker);

    switch (m_state.update) {
    case UpdateState::Idle:
    case UpdateState::Scheduled:
        return;
    case UpdateState::InProgress:
        if (m_state.pendingUpdate) {
            m_state.pendingUpdate = false;
            m_state.update = UpdateState::Scheduled;
            if (!m_state.isSuspended)
                m_updateTimer.startOneShot(0_s);
            return;
        }

        m_state.update = UpdateState::Idle;
        return;
    }
}

void CompositingRunLoop::updateTimerFired()
{
    {
        // Scene update is now in progress.
        Locker locker { m_state.lock };
        if (m_state.isSuspended)
            return;
        m_state.update = UpdateState::InProgress;
    }
    m_updateFunction();
}

} // namespace WebKit

#endif // USE(COORDINATED_GRAPHICS)
