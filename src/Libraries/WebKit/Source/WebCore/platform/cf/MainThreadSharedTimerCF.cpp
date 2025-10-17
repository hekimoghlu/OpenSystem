/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#import "config.h"
#import "MainThreadSharedTimer.h"

#include <wtf/AutodrainedPool.h>

#if PLATFORM(MAC)
#import "PowerObserverMac.h"
#elif PLATFORM(IOS_FAMILY)
#import "WebCoreThreadInternal.h"
#import "WebCoreThreadRun.h"
#endif

namespace WebCore {

static RetainPtr<CFRunLoopTimerRef>& sharedTimer()
{
    static NeverDestroyed<RetainPtr<CFRunLoopTimerRef>> sharedTimer;
    return sharedTimer;
}
static void timerFired(CFRunLoopTimerRef, void*);

static const CFTimeInterval kCFTimeIntervalDistantFuture = std::numeric_limits<CFTimeInterval>::max();

bool& MainThreadSharedTimer::shouldSetupPowerObserver()
{
    static bool setup = true;
    return setup;
}

#if PLATFORM(IOS_FAMILY)
static void applicationDidBecomeActive(CFNotificationCenterRef, void*, CFStringRef, const void*, CFDictionaryRef)
{
    WebThreadRun(^{
        MainThreadSharedTimer::restartSharedTimer();
    });
}
#endif

static void setupPowerObserver()
{
    if (!MainThreadSharedTimer::shouldSetupPowerObserver())
        return;
#if PLATFORM(MAC)
    static PowerObserver* powerObserver;
    if (!powerObserver)
        powerObserver = makeUnique<PowerObserver>(MainThreadSharedTimer::restartSharedTimer).release();
#elif PLATFORM(IOS_FAMILY)
    static bool registeredForApplicationNotification = false;
    if (!registeredForApplicationNotification) {
        registeredForApplicationNotification = true;
        CFNotificationCenterRef notificationCenter = CFNotificationCenterGetLocalCenter();
        CFNotificationCenterAddObserver(notificationCenter, nullptr, applicationDidBecomeActive, CFSTR("UIApplicationDidBecomeActiveNotification"), nullptr, CFNotificationSuspensionBehaviorCoalesce);
    }
#endif
}

static void timerFired(CFRunLoopTimerRef, void*)
{
    AutodrainedPool pool;
    MainThreadSharedTimer::singleton().fired();
}

void MainThreadSharedTimer::restartSharedTimer()
{
    if (!sharedTimer())
        return;

    MainThreadSharedTimer::singleton().stop();
    timerFired(0, 0);
}

void MainThreadSharedTimer::invalidate()
{
    if (!sharedTimer())
        return;

    CFRunLoopTimerInvalidate(sharedTimer().get());
    sharedTimer() = nullptr;
}

void MainThreadSharedTimer::setFireInterval(Seconds interval)
{
    ASSERT(m_firedFunction);

    CFAbsoluteTime fireDate = CFAbsoluteTimeGetCurrent() + interval.value();
    if (!sharedTimer()) {
        sharedTimer() = adoptCF(CFRunLoopTimerCreate(nullptr, fireDate, kCFTimeIntervalDistantFuture, 0, 0, timerFired, nullptr));
#if PLATFORM(IOS_FAMILY)
        CFRunLoopAddTimer(WebThreadRunLoop(), sharedTimer().get(), kCFRunLoopCommonModes);
#else
        CFRunLoopAddTimer(CFRunLoopGetCurrent(), sharedTimer().get(), kCFRunLoopCommonModes);
#endif

        setupPowerObserver();

        return;
    }

    CFRunLoopTimerSetNextFireDate(sharedTimer().get(), fireDate);
}

void MainThreadSharedTimer::stop()
{
    if (!sharedTimer())
        return;

    CFRunLoopTimerSetNextFireDate(sharedTimer().get(), kCFTimeIntervalDistantFuture);
}

} // namespace WebCore
