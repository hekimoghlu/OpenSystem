/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
// Borrowed from Chromium's src/base/numerics/safe_conversions.h.

#ifndef RTC_BASE_NUMERICS_SAFE_CONVERSIONS_H_
#define RTC_BASE_NUMERICS_SAFE_CONVERSIONS_H_

#include <limits>

#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions_impl.h"

namespace rtc {

// Convenience function that returns true if the supplied value is in range
// for the destination type.
template <typename Dst, typename Src>
inline constexpr bool IsValueInRangeForNumericType(Src value) {
  return internal::RangeCheck<Dst>(value) == internal::TYPE_VALID;
}

// checked_cast<> and dchecked_cast<> are analogous to static_cast<> for
// numeric types, except that they [D]CHECK that the specified numeric
// conversion will not overflow or underflow. NaN source will always trigger
// the [D]CHECK.
template <typename Dst, typename Src>
inline constexpr Dst checked_cast(Src value) {
  RTC_CHECK(IsValueInRangeForNumericType<Dst>(value));
  return static_cast<Dst>(value);
}
template <typename Dst, typename Src>
inline constexpr Dst dchecked_cast(Src value) {
  RTC_DCHECK(IsValueInRangeForNumericType<Dst>(value));
  return static_cast<Dst>(value);
}

// saturated_cast<> is analogous to static_cast<> for numeric types, except
// that the specified numeric conversion will saturate rather than overflow or
// underflow. NaN assignment to an integral will trigger a RTC_CHECK condition.
template <typename Dst, typename Src>
inline constexpr Dst saturated_cast(Src value) {
  // Optimization for floating point values, which already saturate.
  if (std::numeric_limits<Dst>::is_iec559)
    return static_cast<Dst>(value);

  switch (internal::RangeCheck<Dst>(value)) {
    case internal::TYPE_VALID:
      return static_cast<Dst>(value);

    case internal::TYPE_UNDERFLOW:
      return std::numeric_limits<Dst>::min();

    case internal::TYPE_OVERFLOW:
      return std::numeric_limits<Dst>::max();

    // Should fail only on attempting to assign NaN to a saturated integer.
    case internal::TYPE_INVALID:
      RTC_CHECK_NOTREACHED();
  }

  RTC_CHECK_NOTREACHED();
}

}  // namespace rtc

#endif  // RTC_BASE_NUMERICS_SAFE_CONVERSIONS_H_
