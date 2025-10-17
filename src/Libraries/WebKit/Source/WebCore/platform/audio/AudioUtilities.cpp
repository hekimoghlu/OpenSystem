/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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

#if ENABLE(WEB_AUDIO)

#include "AudioUtilities.h"
#include <random>
#include <wtf/HashFunctions.h>
#include <wtf/MathExtras.h>
#include <wtf/WeakRandomNumber.h>

namespace WebCore {

namespace AudioUtilities {

double discreteTimeConstantForSampleRate(double timeConstant, double sampleRate)
{
    return 1 - exp(-1 / (sampleRate * timeConstant));
}

size_t timeToSampleFrame(double time, double sampleRate, SampleFrameRounding rounding)
{
    // To compute the desired frame number, we pretend we're actually running the
    // context at a much higher sample rate (by a factor of |oversampleFactor|).
    // Round this to get the nearest frame number at the higher rate. Then
    // convert back to the original rate to get a new frame number that may not be
    // an integer. Then use the specified |rounding_mode| to round this to the
    // integer frame number that we need.
    //
    // Doing this partially solves the issue where Fs * (k / Fs) != k when doing
    // floating point arithmtic for integer k and Fs is the sample rate. By
    // oversampling and rounding, we'll get k back most of the time.
    //
    // The oversampling factor MUST be a power of two so as not to introduce
    // additional round-off in computing the oversample frame number.
    constexpr double oversampleFactor = 1024;
    double frame = std::round(time * sampleRate * oversampleFactor) / oversampleFactor;

    switch (rounding) {
    case SampleFrameRounding::Nearest:
        frame = std::round(frame);
        break;
    case SampleFrameRounding::Down:
        frame = std::floor(frame);
        break;
    case SampleFrameRounding::Up:
        frame = std::ceil(frame);
        break;
    }

    // Just return the largest possible size_t value if necessary.
    if (frame >= static_cast<double>(std::numeric_limits<size_t>::max()))
        return std::numeric_limits<size_t>::max();

    return static_cast<size_t>(frame);
}

void applyNoise(std::span<float> values, float standardDeviation)
{
    std::random_device device;
    std::mt19937 generator(device());
    std::normal_distribution<float> distribution(1, standardDeviation);

    constexpr auto maximumTableSize = 2048;
    constexpr auto minimumTableSize = 1024;

    size_t unusedCellCount = weakRandomNumber<uint32_t>() % (maximumTableSize - minimumTableSize);
    size_t tableSize = maximumTableSize - unusedCellCount;

    // Avoid heap allocations on the rendering thread.
    std::array<float, maximumTableSize> multipliers;
    multipliers.fill(std::numeric_limits<float>::infinity());

    for (auto& value : values) {
        auto& multiplier = multipliers[DefaultHash<double>::hash(value) % tableSize];
        if (std::isinf(multiplier))
            multiplier = distribution(generator);
        value *= multiplier;
    }
}

} // AudioUtilites

} // WebCore

#endif // ENABLE(WEB_AUDIO)
