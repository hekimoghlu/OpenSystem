/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
#include "FrameRateMonitor.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FrameRateMonitor);

static constexpr Seconds MinimumAverageDuration = 1_s;
static constexpr Seconds MaxQueueDuration = 2_s;
static constexpr unsigned MaxFrameDelayCount = 3;

void FrameRateMonitor::update()
{
    ++m_frameCount;

    auto frameTime = MonotonicTime::now().secondsSinceEpoch().value();
    auto lastFrameTime = m_observedFrameTimeStamps.isEmpty() ? frameTime : m_observedFrameTimeStamps.last();

    if (m_observedFrameRate) {
        auto maxDelay = MaxFrameDelayCount / m_observedFrameRate;
        if ((frameTime - lastFrameTime) > maxDelay)
            m_lateFrameCallback({ MonotonicTime::fromRawSeconds(frameTime), MonotonicTime::fromRawSeconds(lastFrameTime), m_observedFrameRate, m_frameCount });
    }
    m_observedFrameTimeStamps.append(frameTime);
    m_observedFrameTimeStamps.removeAllMatching([&](auto time) {
        return time <= frameTime - MaxQueueDuration.value();
    });

    auto queueDuration = m_observedFrameTimeStamps.last() - m_observedFrameTimeStamps.first();
    if (queueDuration > MinimumAverageDuration.value())
        m_observedFrameRate = (m_observedFrameTimeStamps.size() / queueDuration);
}

}
