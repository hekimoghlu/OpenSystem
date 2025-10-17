/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#ifndef API_NUMERICS_MATH_UTILS_H_
#define API_NUMERICS_MATH_UTILS_H_

#include <limits>
#include <type_traits>

#include "rtc_base/checks.h"

namespace webrtc {
namespace webrtc_impl {
// Given two numbers `x` and `y` such that x >= y, computes the difference
// x - y without causing undefined behavior due to signed overflow.
template <typename T>
typename std::make_unsigned<T>::type unsigned_difference(T x, T y) {
  static_assert(
      std::is_signed<T>::value,
      "Function unsigned_difference is only meaningful for signed types.");
  RTC_DCHECK_GE(x, y);
  typedef typename std::make_unsigned<T>::type unsigned_type;
  // int -> unsigned conversion repeatedly adds UINT_MAX + 1 until the number
  // can be represented as an unsigned. Since we know that the actual
  // difference x - y can be represented as an unsigned, it is sufficient to
  // compute the difference modulo UINT_MAX + 1, i.e using unsigned arithmetic.
  return static_cast<unsigned_type>(x) - static_cast<unsigned_type>(y);
}

// Provide neutral element with respect to min().
// Typically used as an initial value for running minimum.
template <typename T,
          typename std::enable_if<std::numeric_limits<T>::has_infinity>::type* =
              nullptr>
constexpr T infinity_or_max() {
  return std::numeric_limits<T>::infinity();
}

template <typename T,
          typename std::enable_if<
              !std::numeric_limits<T>::has_infinity>::type* = nullptr>
constexpr T infinity_or_max() {
  // Fallback to max().
  return std::numeric_limits<T>::max();
}

// Provide neutral element with respect to max().
// Typically used as an initial value for running maximum.
template <typename T,
          typename std::enable_if<std::numeric_limits<T>::has_infinity>::type* =
              nullptr>
constexpr T minus_infinity_or_min() {
  static_assert(std::is_signed<T>::value, "Unsupported. Please open a bug.");
  return -std::numeric_limits<T>::infinity();
}

template <typename T,
          typename std::enable_if<
              !std::numeric_limits<T>::has_infinity>::type* = nullptr>
constexpr T minus_infinity_or_min() {
  // Fallback to min().
  return std::numeric_limits<T>::min();
}

}  // namespace webrtc_impl
}  // namespace webrtc

#endif  // API_NUMERICS_MATH_UTILS_H_
