/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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
#import "ProcessStateMonitor.h"

#if PLATFORM(IOS_FAMILY)

#import "Logging.h"
#import "NetworkProcessMessages.h"
#import "ProcessAssertion.h"
#import "RunningBoardServicesSPI.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

static constexpr double maxPrepareForSuspensionDelayInSecond = 15;

WTF_MAKE_TZONE_ALLOCATED_IMPL(ProcessStateMonitor);

ProcessStateMonitor::ProcessStateMonitor(Function<void(bool)>&& becomeSuspendedHandler)
    : m_becomeSuspendedHandler(WTFMove(becomeSuspendedHandler))
    , m_suspendTimer(RunLoop::main(), [this] { suspendTimerFired(); })
{
    RELEASE_LOG(ProcessSuspension, "%p - ProcessStateMonitor::ProcessStateMonitor", this);
    RELEASE_ASSERT(RunLoop::isMain());

    m_rbsMonitor = [RBSProcessMonitor monitorWithConfiguration:[weakThis = WeakPtr { *this }](id<RBSProcessMonitorConfiguring> config) mutable {
        RBSProcessStateDescriptor *descriptor = [RBSProcessStateDescriptor descriptor];
        [descriptor setValues:RBSProcessStateValueLegacyAssertions | RBSProcessStateValueModernAssertions];
        [config setStateDescriptor:descriptor];
        [config setPredicates:@[[RBSProcessPredicate predicateMatchingHandle:[RBSProcessHandle currentProcess]]]];
        [config setUpdateHandler:[weakThis = WTFMove(weakThis)](RBSProcessMonitor *monitor, RBSProcessHandle *process, RBSProcessStateUpdate *update) {
            ensureOnMainRunLoop([weakThis] {
                if (weakThis)
                    weakThis->checkRemainingRunTime();
            });
        }];
    }];

    checkRemainingRunTime();
}

ProcessStateMonitor::~ProcessStateMonitor()
{
    RELEASE_LOG(ProcessSuspension, "%p - ProcessStateMonitor::~ProcessStateMonitor", this);
    RELEASE_ASSERT(RunLoop::isMain());

    [m_rbsMonitor.get() invalidate];
    processDidBecomeRunning();
}

void ProcessStateMonitor::processDidBecomeRunning()
{
    if (m_state == State::Running)
        return;

    if (m_state == State::Suspended)
        m_becomeSuspendedHandler(false);
    else {
        ASSERT(m_suspendTimer.isActive());
        m_suspendTimer.stop();
    }

    m_state = State::Running;
}

void ProcessStateMonitor::processWillBeSuspended(Seconds timeout)
{
    if (m_state != State::Running)
        return;

    RELEASE_LOG(ProcessSuspension, "%p - ProcessStateMonitor::processWillBeSuspended starts timer for %fs", this, timeout.value());
    ASSERT(!m_suspendTimer.isActive());
    m_suspendTimer.startOneShot(timeout);
    m_state = State::WillSuspend;
}

void ProcessStateMonitor::processWillBeSuspendedImmediately()
{
    if (m_state == State::Suspended)
        return;

    RELEASE_LOG(ProcessSuspension, "%p - ProcessStateMonitor::processWillBeSuspendedImmediately", this);
    m_suspendTimer.stop();
    m_becomeSuspendedHandler(true);
    m_state = State::Suspended;
}

void ProcessStateMonitor::suspendTimerFired()
{
    if (m_state != State::WillSuspend)
        return;

    RELEASE_LOG(ProcessSuspension, "%p - ProcessStateMonitor::suspendTimerFired", this);
    processWillBeSuspendedImmediately();
}

void ProcessStateMonitor::checkRemainingRunTime()
{
#if !PLATFORM(IOS_FAMILY_SIMULATOR)
    // runTime is meaningful only on device.
    double remainingRunTime = [[[RBSProcessHandle currentProcess] activeLimitations] runTime];
    if (remainingRunTime == RBSProcessTimeLimitationNone)
        return processDidBecomeRunning();

    if (remainingRunTime <= maxPrepareForSuspensionDelayInSecond)
        return processWillBeSuspendedImmediately();

    processWillBeSuspended(Seconds(remainingRunTime - maxPrepareForSuspensionDelayInSecond));
#endif
}

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
