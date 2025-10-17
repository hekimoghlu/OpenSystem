/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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

#include <WebCore/Page.h>
#include <wtf/HashMap.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class WebPageProxy;
class WebProcessPool;

class PerActivityStateCPUUsageSampler {
    WTF_MAKE_TZONE_ALLOCATED(PerActivityStateCPUUsageSampler);
public:
    explicit PerActivityStateCPUUsageSampler(WebProcessPool&);
    ~PerActivityStateCPUUsageSampler();

    void ref() const;
    void deref() const;

    void reportWebContentCPUTime(Seconds cpuTime, WebCore::ActivityStateForCPUSampling);

private:
    void loggingTimerFired();
    RefPtr<WebPageProxy> pageForLogging() const;

    WeakRef<WebProcessPool> m_processPool;
    RunLoop::Timer m_loggingTimer;
    typedef HashMap<WebCore::ActivityStateForCPUSampling, Seconds, WTF::IntHash<WebCore::ActivityStateForCPUSampling>, WTF::StrongEnumHashTraits<WebCore::ActivityStateForCPUSampling>> CPUTimeInActivityStateMap;
    CPUTimeInActivityStateMap m_cpuTimeInActivityState;
    MonotonicTime m_lastCPUTime;
};

} // namespace WebKit
