/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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
#include "AnimationFrameRate.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

static constexpr OptionSet<ThrottlingReason> halfSpeedThrottlingReasons { ThrottlingReason::LowPowerMode, ThrottlingReason::NonInteractedCrossOriginFrame, ThrottlingReason::VisuallyIdle, ThrottlingReason::AggressiveThermalMitigation };

FramesPerSecond framesPerSecondNearestFullSpeed(FramesPerSecond nominalFramesPerSecond)
{
    if (nominalFramesPerSecond <= FullSpeedFramesPerSecond)
        return nominalFramesPerSecond;

    float fullSpeedRatio = nominalFramesPerSecond / FullSpeedFramesPerSecond;
    FramesPerSecond floorSpeed = nominalFramesPerSecond / std::floor(fullSpeedRatio);
    FramesPerSecond ceilSpeed = nominalFramesPerSecond / std::ceil(fullSpeedRatio);

    return fullSpeedRatio - std::floor(fullSpeedRatio) <= 0.5 ? floorSpeed : ceilSpeed;
}

std::optional<FramesPerSecond> preferredFramesPerSecond(OptionSet<ThrottlingReason> reasons, std::optional<FramesPerSecond> nominalFramesPerSecond, bool preferFrameRatesNear60FPS)
{
    if (reasons.contains(ThrottlingReason::OutsideViewport))
        return std::nullopt;

    if (!nominalFramesPerSecond || *nominalFramesPerSecond == FullSpeedFramesPerSecond) {
        // FIXME: handle ThrottlingReason::VisuallyIdle
        if (reasons.containsAny(halfSpeedThrottlingReasons))
            return HalfSpeedThrottlingFramesPerSecond;

        return FullSpeedFramesPerSecond;
    }

    auto framesPerSecond = preferFrameRatesNear60FPS ? framesPerSecondNearestFullSpeed(*nominalFramesPerSecond) : *nominalFramesPerSecond;
    if (reasons.containsAny(halfSpeedThrottlingReasons))
        framesPerSecond /= IntervalThrottlingFactor;

    return framesPerSecond;
}

Seconds preferredFrameInterval(OptionSet<ThrottlingReason> reasons, std::optional<FramesPerSecond> nominalFramesPerSecond, bool preferFrameRatesNear60FPS)
{
    if (reasons.contains(ThrottlingReason::OutsideViewport))
        return AggressiveThrottlingAnimationInterval;

    if (!nominalFramesPerSecond || *nominalFramesPerSecond == FullSpeedFramesPerSecond) {
        // FIXME: handle ThrottlingReason::VisuallyIdle
        if (reasons.containsAny(halfSpeedThrottlingReasons))
            return HalfSpeedThrottlingAnimationInterval;
        return FullSpeedAnimationInterval;
    }

    auto framesPerSecond = preferFrameRatesNear60FPS ? framesPerSecondNearestFullSpeed(*nominalFramesPerSecond) : *nominalFramesPerSecond;
    auto interval = Seconds(1.0 / framesPerSecond);

    if (reasons.containsAny(halfSpeedThrottlingReasons))
        interval *= IntervalThrottlingFactor;

    return interval;
}

FramesPerSecond preferredFramesPerSecondFromInterval(Seconds preferredFrameInterval)
{
    if (preferredFrameInterval == FullSpeedAnimationInterval)
        return FullSpeedFramesPerSecond;

    if (preferredFrameInterval == HalfSpeedThrottlingAnimationInterval)
        return HalfSpeedThrottlingFramesPerSecond;

    return std::round(1 / preferredFrameInterval.seconds());
}

TextStream& operator<<(TextStream& ts, const OptionSet<ThrottlingReason>& reasons)
{
    bool didAppend = false;

    for (auto reason : reasons) {
        if (didAppend)
            ts << "|";
        switch (reason) {
        case ThrottlingReason::VisuallyIdle:
            ts << "VisuallyIdle";
            break;
        case ThrottlingReason::OutsideViewport:
            ts << "OutsideViewport";
            break;
        case ThrottlingReason::LowPowerMode:
            ts << "LowPowerMode";
            break;
        case ThrottlingReason::NonInteractedCrossOriginFrame:
            ts << "NonInteractedCrossOriginFrame";
            break;
        case ThrottlingReason::ThermalMitigation:
            ts << "ThermalMitigation";
            break;
        case ThrottlingReason::AggressiveThermalMitigation:
            ts << "AggressiveThermalMitigation";
            break;
        }
        didAppend = true;
    }

    if (reasons.isEmpty())
        ts << "[Unthrottled]";
    return ts;
}
} // namespace WebCore
