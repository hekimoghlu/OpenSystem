/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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

#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <wtf/AutodrainedPool.h>
#include <wtf/SchedulePair.h>

namespace WTF {

static RetainPtr<CFRunLoopTimerRef> createTimer(Seconds interval, bool repeat, void(*timerFired)(CFRunLoopTimerRef, void*), void* info)
{
    CFRunLoopTimerContext context = { 0, info, 0, 0, 0 };
    Seconds repeatInterval = repeat ? interval : 0_s;
    return adoptCF(CFRunLoopTimerCreate(kCFAllocatorDefault, CFAbsoluteTimeGetCurrent() + interval.seconds(), repeatInterval.seconds(), 0, 0, timerFired, &context));
}

void RunLoop::performWork(void* context)
{
    AutodrainedPool pool;
    static_cast<RunLoop*>(context)->performWork();
}

RunLoop::RunLoop()
    : m_runLoop(CFRunLoopGetCurrent())
{
    CFRunLoopSourceContext context = { 0, this, 0, 0, 0, 0, 0, 0, 0, performWork };
    m_runLoopSource = adoptCF(CFRunLoopSourceCreate(kCFAllocatorDefault, 0, &context));
    CFRunLoopAddSource(m_runLoop.get(), m_runLoopSource.get(), kCFRunLoopCommonModes);
}

RunLoop::~RunLoop()
{
    CFRunLoopSourceInvalidate(m_runLoopSource.get());
}

void RunLoop::wakeUp()
{
    CFRunLoopSourceSignal(m_runLoopSource.get());
    CFRunLoopWakeUp(m_runLoop.get());
}

RunLoop::CycleResult RunLoop::cycle(RunLoopMode mode)
{
    CFTimeInterval timeInterval = 0.05;
    CFRunLoopRunInMode(mode, timeInterval, true);
    return CycleResult::Continue;
}

void RunLoop::run()
{
    AutodrainedPool pool;
    CFRunLoopRun();
}

void RunLoop::stop()
{
    ASSERT(m_runLoop == CFRunLoopGetCurrent());
    CFRunLoopStop(m_runLoop.get());
}

void RunLoop::dispatch(const SchedulePairHashSet& schedulePairs, Function<void()>&& function)
{
    auto timer = createTimer(0_s, false, [] (CFRunLoopTimerRef timer, void* context) {
        AutodrainedPool pool;

        CFRunLoopTimerInvalidate(timer);

        auto function = adopt(static_cast<Function<void()>::Impl*>(context));
        function();
    }, function.leak());

    for (auto& schedulePair : schedulePairs)
        CFRunLoopAddTimer(schedulePair->runLoop(), timer.get(), schedulePair->mode());
}

// RunLoop::Timer

RunLoop::TimerBase::TimerBase(Ref<RunLoop>&& runLoop)
    : m_runLoop(WTFMove(runLoop))
{
}

RunLoop::TimerBase::~TimerBase()
{
    stop();
}

void RunLoop::TimerBase::start(Seconds interval, bool repeat)
{
    if (m_timer) {
        bool canReschedule = !repeat && !CFRunLoopTimerDoesRepeat(m_timer.get()) && CFRunLoopTimerIsValid(m_timer.get());
        if (canReschedule) {
            CFRunLoopTimerSetNextFireDate(m_timer.get(), CFAbsoluteTimeGetCurrent() + interval.seconds());
            return;
        }

        stop();
    }

    m_timer = createTimer(interval, repeat, [] (CFRunLoopTimerRef cfTimer, void* context) {
        AutodrainedPool pool;

        auto timer = static_cast<TimerBase*>(context);
        if (!CFRunLoopTimerDoesRepeat(cfTimer))
            CFRunLoopTimerInvalidate(cfTimer);

        timer->fired();
    }, this);

    CFRunLoopAddTimer(m_runLoop->m_runLoop.get(), m_timer.get(), kCFRunLoopCommonModes);
}

void RunLoop::TimerBase::stop()
{
    if (!m_timer)
        return;
    
    CFRunLoopTimerInvalidate(m_timer.get());
    m_timer = nullptr;
}

bool RunLoop::TimerBase::isActive() const
{
    return m_timer && CFRunLoopTimerIsValid(m_timer.get());
}

Seconds RunLoop::TimerBase::secondsUntilFire() const
{
    if (isActive())
        return std::max<Seconds>(Seconds { CFRunLoopTimerGetNextFireDate(m_timer.get()) - CFAbsoluteTimeGetCurrent() }, 0_s);
    return 0_s;
}

} // namespace WTF
