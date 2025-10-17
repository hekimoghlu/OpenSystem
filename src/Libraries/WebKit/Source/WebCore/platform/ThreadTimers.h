/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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
#ifndef ThreadTimers_h
#define ThreadTimers_h

#include <wtf/MonotonicTime.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

class SharedTimer;
class ThreadTimers;
class TimerBase;

struct ThreadTimerHeapItem;
typedef Vector<RefPtr<ThreadTimerHeapItem>> ThreadTimerHeap;
    
// A collection of timers per thread. Kept in ThreadGlobalData.
class ThreadTimers {
    WTF_MAKE_TZONE_ALLOCATED(ThreadTimers);
    WTF_MAKE_NONCOPYABLE(ThreadTimers);
public:
    ThreadTimers();

    // Fire timers for this length of time, and then quit to let the run loop process user input events.
    static constexpr auto maxDurationOfFiringTimers { 16_ms };

    // On a thread different then main, we should set the thread's instance of the SharedTimer.
    void setSharedTimer(SharedTimer*);

    ThreadTimerHeap& timerHeap() { return m_timerHeap; }

    void updateSharedTimer();
    void fireTimersInNestedEventLoop();
    void breakFireLoopForRenderingUpdate();

    unsigned nextHeapInsertionCount() { return m_currentHeapInsertionOrder++; }

private:
    void sharedTimerFiredInternal();
    void fireTimersInNestedEventLoopInternal();

    ThreadTimerHeap m_timerHeap;
    SharedTimer* m_sharedTimer { nullptr }; // External object, can be a run loop on a worker thread. Normally set/reset by worker thread.
    bool m_firingTimers { false };
    bool m_shouldBreakFireLoopForRenderingUpdate { false };
    unsigned m_currentHeapInsertionOrder { 0 };
    MonotonicTime m_pendingSharedTimerFireTime;
};

struct ThreadTimerHeapItem : ThreadSafeRefCounted<ThreadTimerHeapItem> {
    WTF_MAKE_COMPACT_TZONE_OR_ISO_ALLOCATED(ThreadTimerHeapItem);

public:
    static RefPtr<ThreadTimerHeapItem> create(TimerBase&, MonotonicTime, unsigned);

    bool hasTimer() const { return m_timer; }
    TimerBase& timer();
    void clearTimer();

    ThreadTimerHeap& timerHeap() const;

    unsigned heapIndex() const;
    void setHeapIndex(unsigned newIndex);
    void setNotInHeap() { m_heapIndex = invalidHeapIndex; }

    bool isInHeap() const { return m_heapIndex != invalidHeapIndex; }
    bool isFirstInHeap() const { return !m_heapIndex; }

    MonotonicTime time;
    unsigned insertionOrder { 0 };

private:
    ThreadTimers& m_threadTimers;
    TimerBase* m_timer { nullptr };
    unsigned m_heapIndex { invalidHeapIndex };

    static const unsigned invalidHeapIndex = static_cast<unsigned>(-1);

    ThreadTimerHeapItem(TimerBase&, MonotonicTime, unsigned);
};

inline TimerBase& ThreadTimerHeapItem::timer()
{
    ASSERT(m_timer);
    return *m_timer;
}

inline void ThreadTimerHeapItem::clearTimer()
{
    ASSERT(!isInHeap());
    m_timer = nullptr;
}

inline unsigned ThreadTimerHeapItem::heapIndex() const
{
    ASSERT(m_heapIndex != invalidHeapIndex);
    return static_cast<unsigned>(m_heapIndex);
}

inline void ThreadTimerHeapItem::setHeapIndex(unsigned newIndex)
{
    ASSERT(newIndex != invalidHeapIndex);
    m_heapIndex = newIndex;
}

inline ThreadTimerHeap& ThreadTimerHeapItem::timerHeap() const
{
    return m_threadTimers.timerHeap();
}

}

#endif
