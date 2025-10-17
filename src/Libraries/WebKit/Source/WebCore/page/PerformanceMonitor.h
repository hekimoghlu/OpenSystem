/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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

#include "ActivityState.h"
#include "Timer.h"
#include <wtf/CPUTime.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Page;

class PerformanceMonitor {
    WTF_MAKE_TZONE_ALLOCATED(PerformanceMonitor);
public:
    explicit PerformanceMonitor(Page&);

    void ref() const;
    void deref() const;

    void didStartProvisionalLoad();
    void didFinishLoad();
    void activityStateChanged(OptionSet<ActivityState> oldState, OptionSet<ActivityState> newState);

private:
    void measurePostLoadCPUUsage();
    void measurePostBackgroundingCPUUsage();
    void measurePerActivityStateCPUUsage();
    void measureCPUUsageInActivityState(ActivityStateForCPUSampling);
    void measurePostLoadMemoryUsage();
    void measurePostBackgroundingMemoryUsage();
    void processMayBecomeInactiveTimerFired();
    static void updateProcessStateForMemoryPressure();

    WeakRef<Page> m_page;

    Timer m_postPageLoadCPUUsageTimer;
    std::optional<CPUTime> m_postLoadCPUTime;
    Timer m_postBackgroundingCPUUsageTimer;
    std::optional<CPUTime> m_postBackgroundingCPUTime;
    Timer m_perActivityStateCPUUsageTimer;
    std::optional<CPUTime> m_perActivityStateCPUTime;

    Timer m_postPageLoadMemoryUsageTimer;
    Timer m_postBackgroundingMemoryUsageTimer;

    Timer m_processMayBecomeInactiveTimer;
    bool m_processMayBecomeInactive { true };
};

}
