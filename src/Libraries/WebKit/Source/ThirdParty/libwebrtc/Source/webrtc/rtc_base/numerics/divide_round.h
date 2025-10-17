/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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
#ifndef RTC_BASE_NUMERICS_DIVIDE_ROUND_H_
#define RTC_BASE_NUMERICS_DIVIDE_ROUND_H_

#include <type_traits>

#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_compare.h"

namespace webrtc {

template <typename Dividend, typename Divisor>
inline auto constexpr DivideRoundUp(Dividend dividend, Divisor divisor) {
  static_assert(std::is_integral<Dividend>(), "");
  static_assert(std::is_integral<Divisor>(), "");
  RTC_DCHECK_GE(dividend, 0);
  RTC_DCHECK_GT(divisor, 0);

  auto quotient = dividend / divisor;
  auto remainder = dividend % divisor;
  return quotient + (remainder > 0 ? 1 : 0);
}

template <typename Dividend, typename Divisor>
inline auto constexpr DivideRoundToNearest(Dividend dividend, Divisor divisor) {
  static_assert(std::is_integral<Dividend>(), "");
  static_assert(std::is_integral<Divisor>(), "");
  RTC_DCHECK_GT(divisor, 0);

  if (dividend < Dividend{0}) {
    auto half_of_divisor = divisor / 2;
    auto quotient = dividend / divisor;
    auto remainder = dividend % divisor;
    if (rtc::SafeGt(-remainder, half_of_divisor)) {
      --quotient;
    }
    return quotient;
  }

  auto half_of_divisor = (divisor - 1) / 2;
  auto quotient = dividend / divisor;
  auto remainder = dividend % divisor;
  if (rtc::SafeGt(remainder, half_of_divisor)) {
    ++quotient;
  }
  return quotient;
}

}  // namespace webrtc

#endif  // RTC_BASE_NUMERICS_DIVIDE_ROUND_H_
