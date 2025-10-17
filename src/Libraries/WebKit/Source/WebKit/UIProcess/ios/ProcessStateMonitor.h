/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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

#if PLATFORM(IOS_FAMILY)

#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/RunLoop.h>
#import <wtf/TZoneMalloc.h>

OBJC_CLASS RBSProcessMonitor;

namespace WebKit {

class ProcessStateMonitor : public RefCountedAndCanMakeWeakPtr<ProcessStateMonitor> {
    WTF_MAKE_TZONE_ALLOCATED(ProcessStateMonitor);
public:
    static Ref<ProcessStateMonitor> create(Function<void(bool)>&& becomeSuspendedHandler)
    {
        return adoptRef(*new ProcessStateMonitor(WTFMove(becomeSuspendedHandler)));
    }

    ~ProcessStateMonitor();
    void processWillBeSuspendedImmediately();

private:
    ProcessStateMonitor(Function<void(bool)>&& becomeSuspendedHandler);

    void processDidBecomeRunning();
    void processWillBeSuspended(Seconds timeout);
    void suspendTimerFired();
    void checkRemainingRunTime();

    enum class State : uint8_t { Running, WillSuspend, Suspended };
    State m_state { State::Running };
    Function<void(bool)> m_becomeSuspendedHandler;
    RunLoop::Timer m_suspendTimer;
    RetainPtr<RBSProcessMonitor> m_rbsMonitor;
};

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
