/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#ifndef RTC_BASE_RANDOM_H_
#define RTC_BASE_RANDOM_H_

#include <stdint.h>

#include <limits>

#include "rtc_base/checks.h"

namespace webrtc {

class Random {
 public:
  // TODO(tommi): Change this so that the seed can be initialized internally,
  // e.g. by offering two ways of constructing or offer a static method that
  // returns a seed that's suitable for initialization.
  // The problem now is that callers are calling clock_->TimeInMicroseconds()
  // which calls TickTime::Now().Ticks(), which can return a very low value on
  // Mac and can result in a seed of 0 after conversion to microseconds.
  // Besides the quality of the random seed being poor, this also requires
  // the client to take on extra dependencies to generate a seed.
  // If we go for a static seed generator in Random, we can use something from
  // webrtc/rtc_base and make sure that it works the same way across platforms.
  // See also discussion here: https://codereview.webrtc.org/1623543002/
  explicit Random(uint64_t seed);

  Random() = delete;
  Random(const Random&) = delete;
  Random& operator=(const Random&) = delete;

  // Return pseudo-random integer of the specified type.
  // We need to limit the size to 32 bits to keep the output close to uniform.
  template <typename T>
  T Rand() {
    static_assert(std::numeric_limits<T>::is_integer &&
                      std::numeric_limits<T>::radix == 2 &&
                      std::numeric_limits<T>::digits <= 32,
                  "Rand is only supported for built-in integer types that are "
                  "32 bits or smaller.");
    return static_cast<T>(NextOutput());
  }

  // Uniformly distributed pseudo-random number in the interval [0, t].
  uint32_t Rand(uint32_t t);

  // Uniformly distributed pseudo-random number in the interval [low, high].
  uint32_t Rand(uint32_t low, uint32_t high);

  // Uniformly distributed pseudo-random number in the interval [low, high].
  int32_t Rand(int32_t low, int32_t high);

  // Normal Distribution.
  double Gaussian(double mean, double standard_deviation);

  // Exponential Distribution.
  double Exponential(double lambda);

 private:
  // Outputs a nonzero 64-bit random number using Xorshift algorithm.
  // https://en.wikipedia.org/wiki/Xorshift
  uint64_t NextOutput() {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    RTC_DCHECK(state_ != 0x0ULL);
    return state_ * 2685821657736338717ull;
  }

  uint64_t state_;
};

// Return pseudo-random number in the interval [0.0, 1.0).
template <>
float Random::Rand<float>();

// Return pseudo-random number in the interval [0.0, 1.0).
template <>
double Random::Rand<double>();

// Return pseudo-random boolean value.
template <>
bool Random::Rand<bool>();

}  // namespace webrtc

#endif  // RTC_BASE_RANDOM_H_
