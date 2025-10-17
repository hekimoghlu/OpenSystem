/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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

//===-- language/Compability/Common/bit-population-count.h -------------*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// Author: Tunjay Akbarli
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_COMPABILITY_COMMON_BIT_POPULATION_COUNT_H_
#define LANGUAGE_COMPABILITY_COMMON_BIT_POPULATION_COUNT_H_

// Fast and portable functions that implement Fortran's POPCNT and POPPAR
// intrinsic functions.  POPCNT returns the number of bits that are set (1)
// in its argument.  POPPAR is a parity function that returns true
// when POPCNT is odd.

#include <climits>
#include <type_traits>

namespace language::Compability::common {

template <typename INT,
    std::enable_if_t<(sizeof(INT) > 4 && sizeof(INT) <= 8), int> = 0>
inline constexpr int BitPopulationCount(INT x) {
  // In each of the 32 2-bit fields, count the bits that were present.
  // This leaves a value [0..2] in each of these 2-bit fields.
  x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555);
  // Combine into 16 4-bit fields, each holding [0..4]
  x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
  // Now 8 8-bit fields, each with [0..8] in their lower 4 bits.
  x = (x & 0x0f0f0f0f0f0f0f0f) + ((x >> 4) & 0x0f0f0f0f0f0f0f0f);
  // Now 4 16-bit fields, each with [0..16] in their lower 5 bits.
  x = (x & 0x001f001f001f001f) + ((x >> 8) & 0x001f001f001f001f);
  // Now 2 32-bit fields, each with [0..32] in their lower 6 bits.
  x = (x & 0x0000003f0000003f) + ((x >> 16) & 0x0000003f0000003f);
  // Last step: 1 64-bit field, with [0..64]
  return (x & 0x7f) + (x >> 32);
}

template <typename INT,
    std::enable_if_t<(sizeof(INT) > 2 && sizeof(INT) <= 4), int> = 0>
inline constexpr int BitPopulationCount(INT x) {
  // In each of the 16 2-bit fields, count the bits that were present.
  // This leaves a value [0..2] in each of these 2-bit fields.
  x = (x & 0x55555555) + ((x >> 1) & 0x55555555);
  // Combine into 8 4-bit fields, each holding [0..4]
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  // Now 4 8-bit fields, each with [0..8] in their lower 4 bits.
  x = (x & 0x0f0f0f0f) + ((x >> 4) & 0x0f0f0f0f);
  // Now 2 16-bit fields, each with [0..16] in their lower 5 bits.
  x = (x & 0x001f001f) + ((x >> 8) & 0x001f001f);
  // Last step: 1 32-bit field, with [0..32]
  return (x & 0x3f) + (x >> 16);
}

template <typename INT, std::enable_if_t<sizeof(INT) == 2, int> = 0>
inline constexpr int BitPopulationCount(INT x) {
  // In each of the 8 2-bit fields, count the bits that were present.
  // This leaves a value [0..2] in each of these 2-bit fields.
  x = (x & 0x5555) + ((x >> 1) & 0x5555);
  // Combine into 4 4-bit fields, each holding [0..4]
  x = (x & 0x3333) + ((x >> 2) & 0x3333);
  // Now 2 8-bit fields, each with [0..8] in their lower 4 bits.
  x = (x & 0x0f0f) + ((x >> 4) & 0x0f0f);
  // Last step: 1 16-bit field, with [0..16]
  return (x & 0x1f) + (x >> 8);
}

template <typename INT, std::enable_if_t<sizeof(INT) == 1, int> = 0>
inline constexpr int BitPopulationCount(INT x) {
  // In each of the 4 2-bit fields, count the bits that were present.
  // This leaves a value [0..2] in each of these 2-bit fields.
  x = (x & 0x55) + ((x >> 1) & 0x55);
  // Combine into 2 4-bit fields, each holding [0..4]
  x = (x & 0x33) + ((x >> 2) & 0x33);
  // Last step: 1 8-bit field, with [0..8]
  return (x & 0xf) + (x >> 4);
}

template <typename INT> inline constexpr bool Parity(INT x) {
  return BitPopulationCount(x) & 1;
}

// "Parity is for farmers." -- Seymour R. Cray

template <typename INT> inline constexpr int TrailingZeroBitCount(INT x) {
  if ((x & 1) != 0) {
    return 0; // fast path for odd values
  } else if (x == 0) {
    return CHAR_BIT * sizeof x;
  } else {
    return BitPopulationCount(static_cast<INT>(x ^ (x - 1))) - 1;
  }
}
} // namespace language::Compability::common
#endif // FORTRAN_COMMON_BIT_POPULATION_COUNT_H_
