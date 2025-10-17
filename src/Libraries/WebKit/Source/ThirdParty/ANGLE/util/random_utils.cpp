/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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

//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// random_utils:
//   Helper functions for random number generation.
//

#include "random_utils.h"

#include <chrono>

#include <cstdlib>

namespace angle
{

// Seed from clock
RNG::RNG()
{
    long long timeSeed = std::chrono::system_clock::now().time_since_epoch().count();
    mGenerator.seed(static_cast<unsigned int>(timeSeed));
}

// Seed from fixed number.
RNG::RNG(unsigned int seed) : mGenerator(seed) {}

RNG::~RNG() {}

void RNG::reseed(unsigned int newSeed)
{
    mGenerator.seed(newSeed);
}

bool RNG::randomBool(float probTrue)
{
    std::bernoulli_distribution dist(probTrue);
    return dist(mGenerator);
}

int RNG::randomInt()
{
    std::uniform_int_distribution<int> intDistribution;
    return intDistribution(mGenerator);
}

int RNG::randomIntBetween(int min, int max)
{
    std::uniform_int_distribution<int> intDistribution(min, max);
    return intDistribution(mGenerator);
}

unsigned int RNG::randomUInt()
{
    std::uniform_int_distribution<unsigned int> uintDistribution;
    return uintDistribution(mGenerator);
}

float RNG::randomFloat()
{
    std::uniform_real_distribution<float> floatDistribution;
    return floatDistribution(mGenerator);
}

float RNG::randomFloatBetween(float min, float max)
{
    std::uniform_real_distribution<float> floatDistribution(min, max);
    return floatDistribution(mGenerator);
}

float RNG::randomFloatNonnegative()
{
    std::uniform_real_distribution<float> floatDistribution(0.0f,
                                                            std::numeric_limits<float>::max());
    return floatDistribution(mGenerator);
}

float RNG::randomNegativeOneToOne()
{
    return randomFloatBetween(-1.0f, 1.0f);
}

}  // namespace angle
