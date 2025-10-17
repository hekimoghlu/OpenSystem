/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
#include "OpportunisticTaskScheduler.h"

#include "CommonVM.h"
#include "GCController.h"
#include "IdleCallbackController.h"
#include "Page.h"
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/JSGlobalObject.h>
#include <wtf/DataLog.h>
#include <wtf/SystemTracing.h>

namespace WebCore {

OpportunisticTaskScheduler::OpportunisticTaskScheduler(Page& page)
    : m_page(&page)
    , m_runLoopObserver(makeUnique<RunLoopObserver>(RunLoopObserver::WellKnownOrder::PostRenderingUpdate, [weakThis = WeakPtr { this }] {
        if (auto protectedThis = weakThis.get())
            protectedThis->runLoopObserverFired();
    }, RunLoopObserver::Type::OneShot))
{
}

OpportunisticTaskScheduler::~OpportunisticTaskScheduler() = default;

void OpportunisticTaskScheduler::rescheduleIfNeeded(MonotonicTime deadline)
{
    RefPtr page = m_page.get();
    if (page->isWaitingForLoadToFinish() || !page->isVisibleAndActive())
        return;

    auto hasIdleCallbacks = page->findMatchingLocalDocument([](const Document& document) {
        return document.hasPendingIdleCallback();
    });
    if (!hasIdleCallbacks && !page->settings().opportunisticSweepingAndGarbageCollectionEnabled())
        return;

    m_runloopCountAfterBeingScheduled = 0;
    m_currentDeadline = deadline;
    m_runLoopObserver->invalidate();
    if (!m_runLoopObserver->isScheduled())
        m_runLoopObserver->schedule();
}

Ref<ImminentlyScheduledWorkScope> OpportunisticTaskScheduler::makeScheduledWorkScope()
{
    return ImminentlyScheduledWorkScope::create(*this);
}

void OpportunisticTaskScheduler::runLoopObserverFired()
{
    constexpr bool verbose = false;

    if (!m_currentDeadline)
        return;

#if USE(WEB_THREAD)
    if (WebThreadIsEnabled())
        return;
#endif

    if (UNLIKELY(!m_page))
        return;

    RefPtr page = m_page.get();
    if (page->isWaitingForLoadToFinish())
        return;

    if (!page->isVisibleAndActive())
        return;

    auto currentTime = ApproximateTime::now();
    auto remainingTime = m_currentDeadline.secondsSinceEpoch() - currentTime.secondsSinceEpoch();
    if (remainingTime < 0_ms)
        return;

    m_runloopCountAfterBeingScheduled++;

    bool shouldRunTask = [&] {
        static constexpr auto fractionOfRenderingIntervalWhenScheduledWorkIsImminent = 0.95;
        if (remainingTime > fractionOfRenderingIntervalWhenScheduledWorkIsImminent * page->preferredRenderingUpdateInterval())
            return true;

        static constexpr auto minimumRunloopCountWhenScheduledWorkIsImminent = 4;
        if (m_runloopCountAfterBeingScheduled > minimumRunloopCountWhenScheduledWorkIsImminent)
            return true;

        dataLogLnIf(verbose, "[OPPORTUNISTIC TASK] GaveUp: task does not get scheduled ", remainingTime, " ", hasImminentlyScheduledWork(), " ", page->preferredRenderingUpdateInterval(), " ", m_runloopCountAfterBeingScheduled, " signpost:(", JSC::activeJSGlobalObjectSignpostIntervalCount.load(), ")");
        return false;
    }();

    if (!shouldRunTask) {
        dataLogLnIf(verbose, "[OPPORTUNISTIC TASK] RunLoopObserverInvalidate", " signpost:(", JSC::activeJSGlobalObjectSignpostIntervalCount.load(), ")");
        m_runLoopObserver->invalidate();
        m_runLoopObserver->schedule();
        return;
    }

    TraceScope tracingScope {
        PerformOpportunisticallyScheduledTasksStart,
        PerformOpportunisticallyScheduledTasksEnd,
        static_cast<uint64_t>(remainingTime.microseconds())
    };

    auto deadline = std::exchange(m_currentDeadline, MonotonicTime { });
    page->opportunisticallyRunIdleCallbacks(deadline);

    if (!page->settings().opportunisticSweepingAndGarbageCollectionEnabled()) {
        dataLogLnIf(verbose, "[OPPORTUNISTIC TASK] GaveUp: opportunistic sweep and GC is not enabled", " signpost:(", JSC::activeJSGlobalObjectSignpostIntervalCount.load(), ")");
        return;
    }

    page->performOpportunisticallyScheduledTasks(deadline);
}

ImminentlyScheduledWorkScope::ImminentlyScheduledWorkScope(OpportunisticTaskScheduler& scheduler)
    : m_scheduler(&scheduler)
{
    scheduler.m_imminentlyScheduledWorkCount++;
}

ImminentlyScheduledWorkScope::~ImminentlyScheduledWorkScope()
{
    if (m_scheduler)
        m_scheduler->m_imminentlyScheduledWorkCount--;
}

static bool isBusyForTimerBasedGC(JSC::VM& vm)
{
    bool isVisibleAndActive = false;
    bool hasPendingTasks = false;
    bool opportunisticSweepingAndGarbageCollectionEnabled = false;
    Page::forEachPage([&](auto& page) {
        if (page.isVisibleAndActive())
            isVisibleAndActive = true;
        if (page.isWaitingForLoadToFinish())
            hasPendingTasks = true;
        if (page.opportunisticTaskScheduler().hasImminentlyScheduledWork())
            hasPendingTasks = true;
        if (vm.deferredWorkTimer->hasImminentlyScheduledWork())
            hasPendingTasks = true;
        if (page.settings().opportunisticSweepingAndGarbageCollectionEnabled())
            opportunisticSweepingAndGarbageCollectionEnabled = true;
    });

    // If all pages are not visible, we do not care about this GC tasks. We should just run as requested.
    return opportunisticSweepingAndGarbageCollectionEnabled && isVisibleAndActive && hasPendingTasks;
}

OpportunisticTaskScheduler::FullGCActivityCallback::FullGCActivityCallback(JSC::Heap& heap)
    : Base(heap, JSC::Synchronousness::Sync)
    , m_vm(heap.vm())
    , m_runLoopObserver(makeUnique<RunLoopObserver>(RunLoopObserver::WellKnownOrder::PostRenderingUpdate, [this] {
        JSC::JSLockHolder locker(m_vm);
        m_version = 0;
        m_deferCount = 0;
        Base::doCollection(m_vm);
    }, RunLoopObserver::Type::OneShot))
{
}

// We would like to keep FullGCActivityCallback::doCollection and EdenGCActivityCallback::doCollection separate
// since we would like to encode more and more different heuristics for them.
void OpportunisticTaskScheduler::FullGCActivityCallback::doCollection(JSC::VM& vm)
{
    constexpr Seconds delay { 100_ms };
    constexpr unsigned deferCountThreshold = 3;

    if (isBusyForTimerBasedGC(vm)) {
        if (!m_version || m_version != vm.heap.objectSpace().markingVersion()) {
            m_version = vm.heap.objectSpace().markingVersion();
            m_deferCount = 0;
            m_delay = delay;
            setTimeUntilFire(delay);
            return;
        }

        // deferredWorkTimer->hasImminentlyScheduledWork() typically means a wasm compilation is happening right now so we REALLY don't want to GC now.
        if (++m_deferCount < deferCountThreshold || vm.deferredWorkTimer->hasImminentlyScheduledWork()) {
            m_delay = delay;
            setTimeUntilFire(delay);
            return;
        }

        m_runLoopObserver->invalidate();
        m_runLoopObserver->schedule();
        return;
    }

    JSC::JSLockHolder locker(m_vm);
    m_version = 0;
    m_deferCount = 0;
    Base::doCollection(vm);
}

OpportunisticTaskScheduler::EdenGCActivityCallback::EdenGCActivityCallback(JSC::Heap& heap)
    : Base(heap, JSC::Synchronousness::Sync)
    , m_vm(heap.vm())
    , m_runLoopObserver(makeUnique<RunLoopObserver>(RunLoopObserver::WellKnownOrder::PostRenderingUpdate, [this] {
        JSC::JSLockHolder locker(m_vm);
        m_version = 0;
        m_deferCount = 0;
        Base::doCollection(m_vm);
    }, RunLoopObserver::Type::OneShot))
{
}

void OpportunisticTaskScheduler::EdenGCActivityCallback::doCollection(JSC::VM& vm)
{
    constexpr Seconds delay { 10_ms };
    constexpr unsigned deferCountThreshold = 5;

    if (isBusyForTimerBasedGC(vm)) {
        if (!m_version || m_version != vm.heap.objectSpace().edenVersion()) {
            m_version = vm.heap.objectSpace().edenVersion();
            m_deferCount = 0;
            m_delay = delay;
            setTimeUntilFire(delay);
            return;
        }

        // deferredWorkTimer->hasImminentlyScheduledWork() typically means a wasm compilation is happening right now so we REALLY don't want to GC now.
        if (++m_deferCount < deferCountThreshold || vm.deferredWorkTimer->hasImminentlyScheduledWork()) {
            m_delay = delay;
            setTimeUntilFire(delay);
            return;
        }

        m_runLoopObserver->invalidate();
        m_runLoopObserver->schedule();
        return;
    }

    JSC::JSLockHolder locker(m_vm);
    m_version = 0;
    m_deferCount = 0;
    Base::doCollection(m_vm);
}

} // namespace WebCore
