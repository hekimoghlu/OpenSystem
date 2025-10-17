/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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

#include <wtf/OptionSet.h>
#include <wtf/Seconds.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

using FramesPerSecond = unsigned;

enum class ThrottlingReason : uint8_t {
    VisuallyIdle                    = 1 << 0,
    OutsideViewport                 = 1 << 1,
    LowPowerMode                    = 1 << 2,
    NonInteractedCrossOriginFrame   = 1 << 3,
    ThermalMitigation               = 1 << 4,
    AggressiveThermalMitigation     = 1 << 5,
};

// Allow a little more than 60fps to make sure we can at least hit that frame rate.
constexpr const Seconds FullSpeedAnimationInterval { 15_ms };
// Allow a little more than 30fps to make sure we can at least hit that frame rate.
constexpr const Seconds HalfSpeedThrottlingAnimationInterval { 30_ms };
constexpr const Seconds AggressiveThrottlingAnimationInterval { 10_s };
constexpr const int IntervalThrottlingFactor { 2 };

constexpr const FramesPerSecond FullSpeedFramesPerSecond = 60;
constexpr const FramesPerSecond HalfSpeedThrottlingFramesPerSecond = 30;

WEBCORE_EXPORT FramesPerSecond framesPerSecondNearestFullSpeed(FramesPerSecond);

// This will return std::nullopt if throttling results in a frequency < 1fps.
WEBCORE_EXPORT std::optional<FramesPerSecond> preferredFramesPerSecond(OptionSet<ThrottlingReason>, std::optional<FramesPerSecond> nominalFramesPerSecond, bool preferFrameRatesNear60FPS);

WEBCORE_EXPORT Seconds preferredFrameInterval(OptionSet<ThrottlingReason>, std::optional<FramesPerSecond> nominalFramesPerSecond, bool preferFrameRatesNear60FPS);

WEBCORE_EXPORT FramesPerSecond preferredFramesPerSecondFromInterval(Seconds);

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const OptionSet<ThrottlingReason>&);

} // namespace WebCore
