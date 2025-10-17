/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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

#ifndef UTIL_RANDOM_UTILS_H
#define UTIL_RANDOM_UTILS_H

#include <random>
#include <vector>

#include "common/vector_utils.h"
#include "util/util_export.h"

namespace angle
{

class ANGLE_UTIL_EXPORT RNG
{
  public:
    // Seed from clock
    RNG();
    // Seed from fixed number.
    RNG(unsigned int seed);
    ~RNG();

    void reseed(unsigned int newSeed);

    bool randomBool(float probTrue = 0.5f);
    int randomInt();
    int randomIntBetween(int min, int max);
    unsigned int randomUInt();
    float randomFloat();
    float randomFloatBetween(float min, float max);
    float randomFloatNonnegative();
    float randomNegativeOneToOne();

    template <class T>
    T &randomSelect(std::vector<T> &elements)
    {
        return elements[randomIntBetween(0, static_cast<int>(elements.size()) - 1)];
    }

    template <class T>
    const T &randomSelect(const std::vector<T> &elements)
    {
        return elements.at(randomIntBetween(0, static_cast<int>(elements.size()) - 1));
    }

  private:
    std::default_random_engine mGenerator;
};

// Implemented inline to avoid cross-module allocation issues.
inline void FillVectorWithRandomUBytes(RNG *rng, std::vector<uint8_t> *data)
{
    for (size_t i = 0; i < data->size(); ++i)
    {
        (*data)[i] = static_cast<uint8_t>(rng->randomIntBetween(0, 255));
    }
}

inline void FillVectorWithRandomUBytes(std::vector<uint8_t> *data)
{
    RNG rng;
    FillVectorWithRandomUBytes(&rng, data);
}

inline Vector3 RandomVec3(int seed, float minValue, float maxValue)
{
    RNG rng(seed);
    srand(seed);
    return Vector3(rng.randomFloatBetween(minValue, maxValue),
                   rng.randomFloatBetween(minValue, maxValue),
                   rng.randomFloatBetween(minValue, maxValue));
}

inline Vector4 RandomVec4(int seed, float minValue, float maxValue)
{
    RNG rng(seed);
    srand(seed);
    return Vector4(
        rng.randomFloatBetween(minValue, maxValue), rng.randomFloatBetween(minValue, maxValue),
        rng.randomFloatBetween(minValue, maxValue), rng.randomFloatBetween(minValue, maxValue));
}
}  // namespace angle

#endif  // UTIL_RANDOM_UTILS_H
