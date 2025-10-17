/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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
#import "RunLoopObserver.h"

#import <CoreFoundation/CoreFoundation.h>

namespace WebCore {

static constexpr CFIndex cfRunLoopOrder(RunLoopObserver::WellKnownOrder order)
{
    constexpr auto coreAnimationCommit = 2000000;
    switch (order) {
    case RunLoopObserver::WellKnownOrder::GraphicsCommit:
        return coreAnimationCommit;
    case RunLoopObserver::WellKnownOrder::RenderingUpdate:
        return coreAnimationCommit - 1;
    case RunLoopObserver::WellKnownOrder::ActivityStateChange:
        return coreAnimationCommit - 2;
    case RunLoopObserver::WellKnownOrder::InspectorFrameBegin:
        return 0;
    case RunLoopObserver::WellKnownOrder::InspectorFrameEnd:
        return coreAnimationCommit + 1;
    case RunLoopObserver::WellKnownOrder::PostRenderingUpdate:
        return coreAnimationCommit + 2;
    case RunLoopObserver::WellKnownOrder::DisplayRefreshMonitor:
        return 1;
    }
    ASSERT_NOT_REACHED();
    return coreAnimationCommit;
}

static constexpr CFRunLoopActivity cfRunLoopActivity(OptionSet<RunLoopObserver::Activity> activity)
{
    CFRunLoopActivity cfActivity { 0 };
    if (activity.contains(RunLoopObserver::Activity::BeforeWaiting))
        cfActivity |= kCFRunLoopBeforeWaiting;
    if (activity.contains(RunLoopObserver::Activity::Entry))
        cfActivity |= kCFRunLoopEntry;
    if (activity.contains(RunLoopObserver::Activity::Exit))
        cfActivity |= kCFRunLoopExit;
    if (activity.contains(RunLoopObserver::Activity::AfterWaiting))
        cfActivity |= kCFRunLoopAfterWaiting;
    return cfActivity;
}

void RunLoopObserver::runLoopObserverFired(CFRunLoopObserverRef, CFRunLoopActivity, void* context)
{
    static_cast<RunLoopObserver*>(context)->runLoopObserverFired();
}

void RunLoopObserver::schedule(PlatformRunLoop runLoop, OptionSet<Activity> activity)
{
    if (!runLoop)
        runLoop = CFRunLoopGetCurrent();

    // Make sure we wake up the loop or the observer could be delayed until some other source fires.
    CFRunLoopWakeUp(runLoop);

    if (m_runLoopObserver)
        return;

    CFRunLoopObserverContext context = { 0, this, 0, 0, 0 };
    m_runLoopObserver = adoptCF(CFRunLoopObserverCreate(kCFAllocatorDefault, cfRunLoopActivity(activity), isRepeating(), cfRunLoopOrder(m_order), runLoopObserverFired, &context));

    CFRunLoopAddObserver(runLoop, m_runLoopObserver.get(), kCFRunLoopCommonModes);
}

void RunLoopObserver::invalidate()
{
    if (m_runLoopObserver) {
        CFRunLoopObserverInvalidate(m_runLoopObserver.get());
        m_runLoopObserver = nullptr;   
    }
}

bool RunLoopObserver::isScheduled() const
{
    return m_runLoopObserver;
}

} // namespace WebCore
