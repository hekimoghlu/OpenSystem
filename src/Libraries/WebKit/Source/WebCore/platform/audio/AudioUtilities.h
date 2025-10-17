/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
#ifndef AudioUtilities_h
#define AudioUtilities_h

#include <cmath>

namespace WebCore {

namespace AudioUtilities {

// https://www.w3.org/TR/webaudio/#render-quantum-size
constexpr size_t renderQuantumSize = 128;

// Standard functions for converting to and from decibel values from linear.
float linearToDecibels(float);
float decibelsToLinear(float);

// timeConstant is the time it takes a first-order linear time-invariant system
// to reach the value 1 - 1/e (around 63.2%) given a step input response.
// discreteTimeConstantForSampleRate() will return the discrete time-constant for the specific sampleRate.
double discreteTimeConstantForSampleRate(double timeConstant, double sampleRate);

// How to do rounding when converting time to sample frame.
enum class SampleFrameRounding {
    Nearest,
    Down,
    Up
};

// Convert the time to a sample frame at the given sample rate.
size_t timeToSampleFrame(double time, double sampleRate, SampleFrameRounding = SampleFrameRounding::Nearest);

inline float linearToDecibels(float linear)
{
    ASSERT(linear >= 0);
    return 20 * log10f(linear);
}

inline float decibelsToLinear(float decibels)
{
    return powf(10, 0.05f * decibels);
}

void applyNoise(std::span<float> values, float standardDeviation);

} // AudioUtilites

} // WebCore

#endif // AudioUtilities_h
