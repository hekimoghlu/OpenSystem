/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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

namespace WebCore {

// These values were calculated by performing a linear regression on the CSS weights/widths/slopes and Core Text weights/widths/slopes of San Francisco.
// FIXME: <rdar://problem/31312602> Get the real values from Core Text.
inline float normalizeGXWeight(float value)
{
    return 523.7 * value - 109.3;
}

struct KeyFrameValue {
    float ctWeight;
    float cssWeight;
};

// These values were experimentally gathered from the various named weights of San Francisco.
static constexpr std::array keyframes {
    KeyFrameValue { -0.8, 30 },
    KeyFrameValue { -0.4, 274 },
    KeyFrameValue { 0, 400 },
    KeyFrameValue { 0.23, 510 },
    KeyFrameValue { 0.3, 590 },
    KeyFrameValue { 0.4, 700 },
    KeyFrameValue { 0.56, 860 },
    KeyFrameValue { 0.62, 1000 },
};

inline float normalizeCTWeight(float value)
{
    if (value < keyframes[0].ctWeight)
        return keyframes[0].cssWeight;
    for (size_t i = 0; i < std::size(keyframes) - 1; ++i) {
        auto& before = keyframes[i];
        auto& after = keyframes[i + 1];
        if (value >= before.ctWeight && value <= after.ctWeight) {
            float ratio = (value - before.ctWeight) / (after.ctWeight - before.ctWeight);
            return ratio * (after.cssWeight - before.cssWeight) + before.cssWeight;
        }
    }
    return keyframes[std::size(keyframes) - 1].cssWeight;
}

inline float normalizeSlope(float value)
{
    return value * 300;
}

inline float denormalizeGXWeight(float value)
{
    return (value + 109.3) / 523.7;
}

inline float denormalizeCTWeight(float value)
{
    if (value < keyframes[0].cssWeight)
        return keyframes[0].ctWeight;
    for (size_t i = 0; i < std::size(keyframes) - 1; ++i) {
        auto& before = keyframes[i];
        auto& after = keyframes[i + 1];
        if (value >= before.cssWeight && value <= after.cssWeight) {
            float ratio = (value - before.cssWeight) / (after.cssWeight - before.cssWeight);
            return ratio * (after.ctWeight - before.ctWeight) + before.ctWeight;
        }
    }
    return keyframes[std::size(keyframes) - 1].ctWeight;
}

inline float denormalizeSlope(float value)
{
    return value / 300;
}

inline float denormalizeVariationWidth(float value)
{
    if (value <= 125)
        return value / 100;
    if (value <= 150)
        return (value + 125) / 200;
    return (value + 400) / 400;
}

inline float normalizeVariationWidth(float value)
{
    if (value <= 1.25)
        return value * 100;
    if (value <= 1.375)
        return value * 200 - 125;
    return value * 400 - 400;
}

} // namespace WebCore
