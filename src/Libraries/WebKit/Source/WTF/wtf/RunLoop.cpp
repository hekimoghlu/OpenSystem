/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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
#include <wtf/RunLoop.h>

#include <wtf/NeverDestroyed.h>
#include <wtf/Ref.h>
#include <wtf/StdLibExtras.h>
#include <wtf/threads/BinarySemaphore.h>

namespace WTF {

static RunLoop* s_mainRunLoop;
#if USE(WEB_THREAD)
static RunLoop* s_webRunLoop;
#endif

// Helper class for ThreadSpecificData.
class RunLoop::Holder {
    WTF_MAKE_FAST_ALLOCATED;
public:
    Holder()
        : m_runLoop(adoptRef(*new RunLoop))
    {
    }

    ~Holder()
    {
        m_runLoop->threadWillExit();
    }

    RunLoop& runLoop() { return m_runLoop; }

private:
    Ref<RunLoop> m_runLoop;
};

void RunLoop::initializeMain()
{
    RELEASE_ASSERT(!s_mainRunLoop);
    s_mainRunLoop = &RunLoop::current();
}

auto RunLoop::runLoopHolder() -> ThreadSpecific<Holder>&
{
    static LazyNeverDestroyed<ThreadSpecific<Holder>> runLoopHolder;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        runLoopHolder.construct();
    });
    return runLoopHolder;
}

RunLoop& RunLoop::current()
{
    return runLoopHolder()->runLoop();
}

RunLoop& RunLoop::main()
{
    ASSERT(s_mainRunLoop);
    return *s_mainRunLoop;
}

#if USE(WEB_THREAD)
void RunLoop::initializeWeb()
{
    RELEASE_ASSERT(!s_webRunLoop);
    s_webRunLoop = &RunLoop::current();
}

RunLoop& RunLoop::web()
{
    ASSERT(s_webRunLoop);
    return *s_webRunLoop;
}

RunLoop* RunLoop::webIfExists()
{
    return s_webRunLoop;
}
#endif

Ref<RunLoop> RunLoop::create(ASCIILiteral threadName, ThreadType threadType, Thread::QOS qos)
{
    RunLoop* runLoop = nullptr;
    BinarySemaphore semaphore;
    Thread::create(threadName, [&] {
        runLoop = &RunLoop::current();
        semaphore.signal();
        runLoop->run();
    }, threadType, qos)->detach();
    semaphore.wait();
    return *runLoop;
}

bool RunLoop::isCurrent() const
{
    // Avoid constructing the RunLoop for the current thread if it has not been created yet.
    return runLoopHolder().isSet() && this == &RunLoop::current();
}

void RunLoop::performWork()
{
    bool didSuspendFunctions = false;

    {
        Locker locker { m_nextIterationLock };

        // If the RunLoop re-enters or re-schedules, we're expected to execute all functions in order.
        while (!m_currentIteration.isEmpty())
            m_nextIteration.prepend(m_currentIteration.takeLast());

        m_currentIteration = std::exchange(m_nextIteration, { });
    }

    while (!m_currentIteration.isEmpty()) {
        if (m_isFunctionDispatchSuspended) {
            didSuspendFunctions = true;
            break;
        }

        auto function = m_currentIteration.takeFirst();
        function();
    }

    // Suspend only for a single cycle.
    m_isFunctionDispatchSuspended = false;
    m_hasSuspendedFunctions = didSuspendFunctions;

    if (m_hasSuspendedFunctions)
        wakeUp();
}

void RunLoop::dispatch(Function<void()>&& function)
{
    RELEASE_ASSERT(function);
    bool needsWakeup = false;

    {
        Locker locker { m_nextIterationLock };
        needsWakeup = m_nextIteration.isEmpty();
        m_nextIteration.append(WTFMove(function));
    }

    if (needsWakeup)
        wakeUp();
}

Ref<RunLoop::DispatchTimer> RunLoop::dispatchAfter(Seconds delay, Function<void()>&& function)
{
    RELEASE_ASSERT(function);
    Ref<DispatchTimer> timer = adoptRef(*new DispatchTimer(*this));
    timer->setFunction([timer = timer.copyRef(), function = WTFMove(function)]() mutable {
        Ref<DispatchTimer> protectedTimer { WTFMove(timer) };
        function();
        protectedTimer->stop();
    });
    timer->startOneShot(delay);
    return timer;
}

void RunLoop::suspendFunctionDispatchForCurrentCycle()
{
    // Don't suspend if there are already suspended functions to avoid unexecuted function pile-up.
    if (m_isFunctionDispatchSuspended || m_hasSuspendedFunctions)
        return;

    m_isFunctionDispatchSuspended = true;
    // Wake up (even if there is nothing to do) to disable suspension.
    wakeUp();
}

void RunLoop::threadWillExit()
{
    m_currentIteration.clear();
    {
        Locker locker { m_nextIterationLock };
        m_nextIteration.clear();
    }
}

} // namespace WTF
