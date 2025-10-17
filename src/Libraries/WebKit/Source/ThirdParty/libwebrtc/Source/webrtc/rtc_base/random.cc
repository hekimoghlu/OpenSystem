/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
#include "rtc_base/random.h"

#include <math.h>

#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"

namespace webrtc {

Random::Random(uint64_t seed) {
  RTC_DCHECK(seed != 0x0ull);
  state_ = seed;
}

uint32_t Random::Rand(uint32_t t) {
  // Casting the output to 32 bits will give an almost uniform number.
  // Pr[x=0] = (2^32-1) / (2^64-1)
  // Pr[x=k] = 2^32 / (2^64-1) for k!=0
  // Uniform would be Pr[x=k] = 2^32 / 2^64 for all 32-bit integers k.
  uint32_t x = NextOutput();
  // If x / 2^32 is uniform on [0,1), then x / 2^32 * (t+1) is uniform on
  // the interval [0,t+1), so the integer part is uniform on [0,t].
  uint64_t result = x * (static_cast<uint64_t>(t) + 1);
  result >>= 32;
  return result;
}

uint32_t Random::Rand(uint32_t low, uint32_t high) {
  RTC_DCHECK(low <= high);
  return Rand(high - low) + low;
}

int32_t Random::Rand(int32_t low, int32_t high) {
  RTC_DCHECK(low <= high);
  const int64_t low_i64{low};
  return rtc::dchecked_cast<int32_t>(
      Rand(rtc::dchecked_cast<uint32_t>(high - low_i64)) + low_i64);
}

template <>
float Random::Rand<float>() {
  double result = NextOutput() - 1;
  result = result / static_cast<double>(0xFFFFFFFFFFFFFFFFull);
  return static_cast<float>(result);
}

template <>
double Random::Rand<double>() {
  double result = NextOutput() - 1;
  result = result / static_cast<double>(0xFFFFFFFFFFFFFFFFull);
  return result;
}

template <>
bool Random::Rand<bool>() {
  return Rand(0, 1) == 1;
}

double Random::Gaussian(double mean, double standard_deviation) {
  // Creating a Normal distribution variable from two independent uniform
  // variables based on the Box-Muller transform, which is defined on the
  // interval (0, 1]. Note that we rely on NextOutput to generate integers
  // in the range [1, 2^64-1]. Normally this behavior is a bit frustrating,
  // but here it is exactly what we need.
  const double kPi = 3.14159265358979323846;
  double u1 = static_cast<double>(NextOutput()) /
              static_cast<double>(0xFFFFFFFFFFFFFFFFull);
  double u2 = static_cast<double>(NextOutput()) /
              static_cast<double>(0xFFFFFFFFFFFFFFFFull);
  return mean + standard_deviation * sqrt(-2 * log(u1)) * cos(2 * kPi * u2);
}

double Random::Exponential(double lambda) {
  double uniform = Rand<double>();
  return -log(uniform) / lambda;
}

}  // namespace webrtc
