/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
#ifndef AOM_TEST_ACM_RANDOM_H_
#define AOM_TEST_ACM_RANDOM_H_

#include "gtest/gtest.h"

#include "aom/aom_integer.h"

namespace libaom_test {

class ACMRandom {
 public:
  ACMRandom() : random_(DeterministicSeed()) {}

  explicit ACMRandom(int seed) : random_(seed) {}

  void Reset(int seed) { random_.Reseed(seed); }

  // Generates a random 31-bit unsigned integer from [0, 2^31).
  uint32_t Rand31() {
    return random_.Generate(testing::internal::Random::kMaxRange);
  }

  uint16_t Rand16() {
    const uint32_t value =
        random_.Generate(testing::internal::Random::kMaxRange);
    // There's a bit more entropy in the upper bits of this implementation.
    return (value >> 15) & 0xffff;
  }

  int16_t Rand16Signed() { return static_cast<int16_t>(Rand16()); }

  int16_t Rand15() {
    const uint32_t value =
        random_.Generate(testing::internal::Random::kMaxRange);
    // There's a bit more entropy in the upper bits of this implementation.
    return (value >> 16) & 0x7fff;
  }

  int16_t Rand15Signed() {
    // Use 15 bits: values between 16383 (0x3FFF) and -16384 (0xC000).
    return static_cast<int16_t>(Rand15()) - (1 << 14);
  }

  uint16_t Rand12() {
    const uint32_t value =
        random_.Generate(testing::internal::Random::kMaxRange);
    // There's a bit more entropy in the upper bits of this implementation.
    return (value >> 19) & 0xfff;
  }

  uint8_t Rand8() {
    const uint32_t value =
        random_.Generate(testing::internal::Random::kMaxRange);
    // There's a bit more entropy in the upper bits of this implementation.
    return (value >> 23) & 0xff;
  }

  uint8_t Rand8Extremes() {
    // Returns a random value near 0 or near 255, to better exercise
    // saturation behavior.
    const uint8_t r = Rand8();
    return static_cast<uint8_t>((r < 128) ? r << 4 : r >> 4);
  }

  int PseudoUniform(int range) { return random_.Generate(range); }

  int operator()(int n) { return PseudoUniform(n); }

  static int DeterministicSeed() { return 0xbaba; }

 private:
  testing::internal::Random random_;
};

}  // namespace libaom_test

#endif  // AOM_TEST_ACM_RANDOM_H_
