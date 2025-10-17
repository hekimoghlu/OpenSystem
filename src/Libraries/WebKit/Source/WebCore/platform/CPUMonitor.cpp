/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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
#include "CPUMonitor.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CPUMonitor);

CPUMonitor::CPUMonitor(Seconds checkInterval, ExceededCPULimitHandler&& exceededCPULimitHandler)
    : m_checkInterval(checkInterval)
    , m_exceededCPULimitHandler(WTFMove(exceededCPULimitHandler))
    , m_timer(*this, &CPUMonitor::timerFired)
{
}

void CPUMonitor::setCPULimit(const std::optional<double>& cpuLimit)
{
    if (m_cpuLimit == cpuLimit)
        return;

    m_cpuLimit = cpuLimit;
    if (m_cpuLimit) {
        if (!m_timer.isActive()) {
            m_lastCPUTime = CPUTime::get();
            m_timer.startRepeating(m_checkInterval);
        }
    } else
        m_timer.stop();
}

void CPUMonitor::timerFired()
{
    ASSERT(m_cpuLimit);

    if (!m_lastCPUTime) {
        m_lastCPUTime = CPUTime::get();
        return;
    }

    auto cpuTime = CPUTime::get();
    if (!cpuTime)
        return;

    auto cpuUsagePercent = cpuTime.value().percentageCPUUsageSince(m_lastCPUTime.value());
    if (cpuUsagePercent > m_cpuLimit.value() * 100)
        m_exceededCPULimitHandler(cpuUsagePercent / 100.);

    m_lastCPUTime = cpuTime;
}

} // namespace WebCore
