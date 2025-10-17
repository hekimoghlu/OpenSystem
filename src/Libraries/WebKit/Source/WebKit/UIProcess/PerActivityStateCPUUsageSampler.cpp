/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
#include "PerActivityStateCPUUsageSampler.h"

#include "Logging.h"
#include "WebPageProxy.h"
#include "WebProcessPool.h"
#include "WebProcessProxy.h"
#include <WebCore/DiagnosticLoggingClient.h>
#include <WebCore/DiagnosticLoggingKeys.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

static const Seconds loggingInterval { 60_min };

WTF_MAKE_TZONE_ALLOCATED_IMPL(PerActivityStateCPUUsageSampler);

PerActivityStateCPUUsageSampler::PerActivityStateCPUUsageSampler(WebProcessPool& processPool)
    : m_processPool(processPool)
    , m_loggingTimer(RunLoop::main(), this, &PerActivityStateCPUUsageSampler::loggingTimerFired)
{
    m_lastCPUTime = MonotonicTime::now();
    m_loggingTimer.startRepeating(loggingInterval);
}

PerActivityStateCPUUsageSampler::~PerActivityStateCPUUsageSampler() = default;

void PerActivityStateCPUUsageSampler::ref() const
{
    m_processPool->ref();
}

void PerActivityStateCPUUsageSampler::deref() const
{
    m_processPool->deref();
}

void PerActivityStateCPUUsageSampler::reportWebContentCPUTime(Seconds cpuTime, ActivityStateForCPUSampling activityState)
{
    auto result = m_cpuTimeInActivityState.add(activityState, cpuTime);
    if (!result.isNewEntry)
        result.iterator->value += cpuTime;
}

static inline String loggingKeyForActivityState(ActivityStateForCPUSampling state)
{
    switch (state) {
    case ActivityStateForCPUSampling::NonVisible:
        return DiagnosticLoggingKeys::nonVisibleStateKey();
    case ActivityStateForCPUSampling::VisibleNonActive:
        return DiagnosticLoggingKeys::visibleNonActiveStateKey();
    case ActivityStateForCPUSampling::VisibleAndActive:
        return DiagnosticLoggingKeys::visibleAndActiveStateKey();
    }
}

void PerActivityStateCPUUsageSampler::loggingTimerFired()
{
    auto page = pageForLogging();
    if (!page) {
        m_cpuTimeInActivityState.clear();
        return;
    }

    MonotonicTime currentCPUTime = MonotonicTime::now();
    Seconds cpuTimeDelta = currentCPUTime - m_lastCPUTime;

    for (auto& pair : m_cpuTimeInActivityState) {
        double cpuUsage = pair.value.value() * 100. / cpuTimeDelta.value();
        String activityStateKey = loggingKeyForActivityState(pair.key);
        page->logDiagnosticMessageWithValue(DiagnosticLoggingKeys::cpuUsageKey(), activityStateKey, cpuUsage, 2, ShouldSample::No);
        RELEASE_LOG(PerformanceLogging, "WebContent processes used %.1f%% CPU in %s state", cpuUsage, activityStateKey.utf8().data());
    }

    m_cpuTimeInActivityState.clear();
    m_lastCPUTime = currentCPUTime;
}

RefPtr<WebPageProxy> PerActivityStateCPUUsageSampler::pageForLogging() const
{
    for (Ref webProcess : Ref { m_processPool.get() }->processes()) {
        if (!webProcess->pageCount())
            continue;
        return webProcess->pages()[0].ptr();
    }
    return nullptr;
}

} // namespace WebKit
