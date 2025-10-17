/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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
#include "FrameRateAligner.h"

namespace WebCore {
DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(FrameRateAligner);

FrameRateAligner::FrameRateAligner() = default;

FrameRateAligner::~FrameRateAligner() = default;

static ReducedResolutionSeconds idealTimeForNextUpdate(ReducedResolutionSeconds firstUpdateTime, ReducedResolutionSeconds lastUpdateTime, FramesPerSecond frameRate)
{
    ReducedResolutionSeconds interval(1.0 / frameRate);
    auto timeUntilNextUpdate = (lastUpdateTime - firstUpdateTime) % interval;
    return lastUpdateTime + interval - timeUntilNextUpdate;
}

void FrameRateAligner::beginUpdate(ReducedResolutionSeconds timestamp, std::optional<FramesPerSecond> timelineFrameRate)
{
    // We record the timestamp for this new update such that in updateFrameRate()
    // we can compare against it to identify animations that should be sampled
    // for this current update.
    m_timestamp = timestamp;

    const auto nextUpdateTimeEpsilon = 1_ms;
    for (auto& [frameRate, data] : m_frameRates) {
        // We can reset isNew to false for all entries since they were already present
        // in the previous update.
        data.isNew = false;

        // If the timeline frame rate is the same as this animation frame rate, then
        // we don't need to compute the next ideal sample time.
        if (timelineFrameRate == frameRate) {
            data.lastUpdateTime = timestamp;
            continue;
        }

        // If the next ideal sample time for this frame rate aligns with the current timestamp
        // or is already behind the current timestamp, we can set the last update time to the
        // current timestamp, which will indicate in updateFrameRate() that animations using
        // this frame rate should be sampled in the current update.
        auto nextUpdateTime = idealTimeForNextUpdate(data.firstUpdateTime, data.lastUpdateTime, frameRate);
        if ((nextUpdateTime - nextUpdateTimeEpsilon) <= timestamp)
            data.lastUpdateTime = timestamp;
    }
}

auto FrameRateAligner::updateFrameRate(FramesPerSecond frameRate) -> ShouldUpdate
{
    auto it = m_frameRates.find(frameRate);

    if (it != m_frameRates.end()) {
        // We're dealing with a frame rate for which we've sampled animations before.
        // If the last update time for this frame rate is the current timestamp, this
        // means we've established in beginUpdate() that animations for this frame rate
        // should be sampled.
        return it->value.lastUpdateTime == m_timestamp ? ShouldUpdate::Yes : ShouldUpdate::No;
    }

    // We're dealing with a frame rate we didn't see in the previous update. In this case,
    // we'll allow animations to be sampled right away. Later, in finishUpdate(), we'll
    // make sure to reset the last update time to align this new frame rate with other
    // compatible frame rates.
    m_frameRates.set(frameRate, FrameRateData { m_timestamp, m_timestamp });
    return ShouldUpdate::Yes;
}

// For two frame rates to be aligned, one must be the multitple of the other, or vice versa.
static bool frameRatesCanBeAligned(FramesPerSecond a, FramesPerSecond b)
{
    return (a > b && a % b == 0) || (b > a && b % a == 0);
}

void FrameRateAligner::finishUpdate()
{
    // Iterate through the frame rates to find new entries and set their first update time
    // in a way that future updates will be synchronized with other animations with that
    // frame rate.
    for (auto& [frameRate, data] : m_frameRates) {
        if (!data.isNew)
            continue;

        // Look for the compatible frame rate with the highest value.
        std::optional<FramesPerSecond> highestCompatibleFrameRate;
        for (auto& [potentiallyCompatibleFrameRate, potentiallyCompatibleData] : m_frameRates) {
            if (potentiallyCompatibleData.isNew)
                continue;

            if (frameRatesCanBeAligned(frameRate, potentiallyCompatibleFrameRate)) {
                if (!highestCompatibleFrameRate || *highestCompatibleFrameRate > potentiallyCompatibleFrameRate)
                    highestCompatibleFrameRate = potentiallyCompatibleFrameRate;
            }
        }

        // If we don't find any compatible frame rate, we can leave the last update time as-is
        // and use the current timestamp as the basis from which we'll align animations for this
        // frame rate.
        if (highestCompatibleFrameRate)
            data.firstUpdateTime = m_frameRates.get(*highestCompatibleFrameRate).firstUpdateTime;
    }
}

std::optional<Seconds> FrameRateAligner::timeUntilNextUpdateForFrameRate(FramesPerSecond frameRate, ReducedResolutionSeconds timestamp) const
{
    auto it = m_frameRates.find(frameRate);
    if (it == m_frameRates.end())
        return std::nullopt;

    auto& data = it->value;
    return idealTimeForNextUpdate(data.firstUpdateTime, data.lastUpdateTime, frameRate) - timestamp;
}

std::optional<FramesPerSecond> FrameRateAligner::maximumFrameRate() const
{
    std::optional<FramesPerSecond> maximumFrameRate;
    for (auto frameRate : m_frameRates.keys()) {
        if (!maximumFrameRate || *maximumFrameRate < frameRate)
            maximumFrameRate = frameRate;
    }
    return maximumFrameRate;
}

} // namespace WebCore

