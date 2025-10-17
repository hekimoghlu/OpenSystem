/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
#ifndef RTC_BASE_NUMERICS_MOD_OPS_H_
#define RTC_BASE_NUMERICS_MOD_OPS_H_

#include <algorithm>
#include <type_traits>

#include "rtc_base/checks.h"

namespace webrtc {

template <unsigned long M>                                    // NOLINT
inline unsigned long Add(unsigned long a, unsigned long b) {  // NOLINT
  RTC_DCHECK_LT(a, M);
  unsigned long t = M - b % M;  // NOLINT
  unsigned long res = a - t;    // NOLINT
  if (t > a)
    return res + M;
  return res;
}

template <unsigned long M>                                         // NOLINT
inline unsigned long Subtract(unsigned long a, unsigned long b) {  // NOLINT
  RTC_DCHECK_LT(a, M);
  unsigned long sub = b % M;  // NOLINT
  if (a < sub)
    return M - (sub - a);
  return a - sub;
}

// Calculates the forward difference between two wrapping numbers.
//
// Example:
// uint8_t x = 253;
// uint8_t y = 2;
//
// ForwardDiff(x, y) == 5
//
//   252   253   254   255    0     1     2     3
// #################################################
// |     |  x  |     |     |     |     |  y  |     |
// #################################################
//          |----->----->----->----->----->
//
// ForwardDiff(y, x) == 251
//
//   252   253   254   255    0     1     2     3
// #################################################
// |     |  x  |     |     |     |     |  y  |     |
// #################################################
// -->----->                              |----->---
//
// If M > 0 then wrapping occurs at M, if M == 0 then wrapping occurs at the
// largest value representable by T.
template <typename T, T M>
inline typename std::enable_if<(M > 0), T>::type ForwardDiff(T a, T b) {
  static_assert(std::is_unsigned<T>::value,
                "Type must be an unsigned integer.");
  RTC_DCHECK_LT(a, M);
  RTC_DCHECK_LT(b, M);
  return a <= b ? b - a : M - (a - b);
}

template <typename T, T M>
inline typename std::enable_if<(M == 0), T>::type ForwardDiff(T a, T b) {
  static_assert(std::is_unsigned<T>::value,
                "Type must be an unsigned integer.");
  return b - a;
}

template <typename T>
inline T ForwardDiff(T a, T b) {
  return ForwardDiff<T, 0>(a, b);
}

// Calculates the reverse difference between two wrapping numbers.
//
// Example:
// uint8_t x = 253;
// uint8_t y = 2;
//
// ReverseDiff(y, x) == 5
//
//   252   253   254   255    0     1     2     3
// #################################################
// |     |  x  |     |     |     |     |  y  |     |
// #################################################
//          <-----<-----<-----<-----<-----|
//
// ReverseDiff(x, y) == 251
//
//   252   253   254   255    0     1     2     3
// #################################################
// |     |  x  |     |     |     |     |  y  |     |
// #################################################
// ---<-----|                             |<-----<--
//
// If M > 0 then wrapping occurs at M, if M == 0 then wrapping occurs at the
// largest value representable by T.
template <typename T, T M>
inline typename std::enable_if<(M > 0), T>::type ReverseDiff(T a, T b) {
  static_assert(std::is_unsigned<T>::value,
                "Type must be an unsigned integer.");
  RTC_DCHECK_LT(a, M);
  RTC_DCHECK_LT(b, M);
  return b <= a ? a - b : M - (b - a);
}

template <typename T, T M>
inline typename std::enable_if<(M == 0), T>::type ReverseDiff(T a, T b) {
  static_assert(std::is_unsigned<T>::value,
                "Type must be an unsigned integer.");
  return a - b;
}

template <typename T>
inline T ReverseDiff(T a, T b) {
  return ReverseDiff<T, 0>(a, b);
}

// Calculates the minimum distance between to wrapping numbers.
//
// The minimum distance is defined as min(ForwardDiff(a, b), ReverseDiff(a, b))
template <typename T, T M = 0>
inline T MinDiff(T a, T b) {
  static_assert(std::is_unsigned<T>::value,
                "Type must be an unsigned integer.");
  return std::min(ForwardDiff<T, M>(a, b), ReverseDiff<T, M>(a, b));
}

}  // namespace webrtc

#endif  // RTC_BASE_NUMERICS_MOD_OPS_H_
