/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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
#include "VelocityData.h"

#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(HistoricalVelocityData);

VelocityData HistoricalVelocityData::velocityForNewData(FloatPoint newPosition, double scale, MonotonicTime timestamp)
{
    auto append = [&](FloatPoint newPosition, double scale, MonotonicTime timestamp)
    {
        m_latestDataIndex = (m_latestDataIndex + 1) % maxHistoryDepth;
        m_positionHistory[m_latestDataIndex] = { timestamp, newPosition, scale };
        m_historySize = std::min(m_historySize + 1, maxHistoryDepth);
        m_lastAppendTimestamp = timestamp;
    };

    // Due to all the source of rect update, the input is very noisy. To smooth the output, we accumulate all changes
    // within 1 frame as a single update. No speed computation is ever done on data within the same frame.
    const Seconds filteringThreshold(1.0 / 60);

    VelocityData velocityData;
    if (m_historySize > 0) {
        unsigned oldestDataIndex;
        unsigned distanceToLastHistoricalData = m_historySize - 1;
        if (distanceToLastHistoricalData <= m_latestDataIndex)
            oldestDataIndex = m_latestDataIndex - distanceToLastHistoricalData;
        else
            oldestDataIndex = m_historySize - (distanceToLastHistoricalData - m_latestDataIndex);

        Seconds timeDelta = timestamp - m_positionHistory[oldestDataIndex].timestamp;
        if (timeDelta > filteringThreshold) {
            Data& oldestData = m_positionHistory[oldestDataIndex];
            velocityData = VelocityData((newPosition.x() - oldestData.position.x()) / timeDelta.seconds(), (newPosition.y() - oldestData.position.y()) / timeDelta.seconds(), (scale - oldestData.scale) / timeDelta.seconds(), timestamp);
        }
    }

    Seconds timeSinceLastAppend = timestamp - m_lastAppendTimestamp;
    if (timeSinceLastAppend > filteringThreshold)
        append(newPosition, scale, timestamp);
    else
        m_positionHistory[m_latestDataIndex] = { timestamp, newPosition, scale };

    return velocityData;
}

TextStream& operator<<(TextStream& ts, const VelocityData& velocityData)
{
    ts.dumpProperty("timestamp", velocityData.lastUpdateTime.secondsSinceEpoch().value());
    if (velocityData.horizontalVelocity)
        ts.dumpProperty("horizontalVelocity", velocityData.horizontalVelocity);
    if (velocityData.verticalVelocity)
        ts.dumpProperty("verticalVelocity", velocityData.verticalVelocity);
    if (velocityData.scaleChangeRate)
        ts.dumpProperty("scaleChangeRate", velocityData.scaleChangeRate);

    return ts;
}

} // namespace WebCore
