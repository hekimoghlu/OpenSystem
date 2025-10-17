/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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
#include "ThreadTimers.h"

#include "MainThreadSharedTimer.h"
#include "SharedTimer.h"
#include "ThreadGlobalData.h"
#include "Timer.h"
#include <wtf/ApproximateTime.h>
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(IOS_FAMILY)
#include "WebCoreThread.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ThreadTimers);

// Timers are created, started and fired on the same thread, and each thread has its own ThreadTimers
// copy to keep the heap and a set of currently firing timers.

ThreadTimers::ThreadTimers()
{
    if (isUIThread())
        setSharedTimer(&MainThreadSharedTimer::singleton());
}

// A worker thread may initialize SharedTimer after some timers are created.
// Also, SharedTimer can be replaced with nullptr before all timers are destroyed.
void ThreadTimers::setSharedTimer(SharedTimer* sharedTimer)
{
    if (m_sharedTimer) {
        m_sharedTimer->setFiredFunction(nullptr);
        m_sharedTimer->stop();
        m_pendingSharedTimerFireTime = MonotonicTime { };
    }
    
    m_sharedTimer = sharedTimer;
    
    if (sharedTimer) {
        m_sharedTimer->setFiredFunction([] { threadGlobalData().threadTimers().sharedTimerFiredInternal(); });
        updateSharedTimer();
    }
}

void ThreadTimers::updateSharedTimer()
{
    if (!m_sharedTimer)
        return;

    while (!m_timerHeap.isEmpty() && !m_timerHeap.first()->hasTimer()) {
        ASSERT_NOT_REACHED();
        TimerBase::heapDeleteNullMin(m_timerHeap);
    }
    ASSERT(m_timerHeap.isEmpty() || m_timerHeap.first()->hasTimer());

    if (m_firingTimers || m_timerHeap.isEmpty()) {
        m_pendingSharedTimerFireTime = MonotonicTime { };
        m_sharedTimer->stop();
    } else {
        MonotonicTime nextFireTime = m_timerHeap.first()->time;
        MonotonicTime currentMonotonicTime = MonotonicTime::now();
        if (m_pendingSharedTimerFireTime) {
            // No need to restart the timer if both the pending fire time and the new fire time are in the past.
            if (m_pendingSharedTimerFireTime <= currentMonotonicTime && nextFireTime <= currentMonotonicTime)
                return;
        } 
        m_pendingSharedTimerFireTime = nextFireTime;
        m_sharedTimer->setFireInterval(std::max(nextFireTime - currentMonotonicTime, 0_s));
    }
}

void ThreadTimers::sharedTimerFiredInternal()
{
    ASSERT(isMainThread() || (!isWebThread() && !isUIThread()));
    // Do a re-entrancy check.
    if (m_firingTimers)
        return;
    m_firingTimers = true;
    m_pendingSharedTimerFireTime = MonotonicTime { };

    auto fireTime = MonotonicTime::now();
    auto timeToQuit = ApproximateTime::now() + maxDurationOfFiringTimers;

    while (!m_timerHeap.isEmpty()) {
        Ref<ThreadTimerHeapItem> item = *m_timerHeap.first();
        ASSERT(item->hasTimer());
        if (!item->hasTimer()) {
            TimerBase::heapDeleteNullMin(m_timerHeap);
            continue;
        }

        if (item->time > fireTime)
            break;

        auto& timer = item->timer();
        Seconds interval = timer.repeatInterval();
        timer.setNextFireTime(interval ? fireTime + interval : MonotonicTime { });

        // Once the timer has been fired, it may be deleted, so do nothing else with it after this point.
        item->timer().fired();

        // Catch the case where the timer asked timers to fire in a nested event loop, or we are over time limit.
        if (!m_firingTimers || timeToQuit < ApproximateTime::now())
            break;

        if (m_shouldBreakFireLoopForRenderingUpdate)
            break;
    }

    m_firingTimers = false;
    m_shouldBreakFireLoopForRenderingUpdate = false;

    updateSharedTimer();
}

void ThreadTimers::fireTimersInNestedEventLoop()
{
    // Reset the reentrancy guard so the timers can fire again.
    m_firingTimers = false;

    if (m_sharedTimer) {
        m_sharedTimer->invalidate();
        m_pendingSharedTimerFireTime = MonotonicTime { };
    }

    updateSharedTimer();
}

void ThreadTimers::breakFireLoopForRenderingUpdate()
{
    if (!m_firingTimers)
        return;
    m_shouldBreakFireLoopForRenderingUpdate = true;
}

} // namespace WebCore

