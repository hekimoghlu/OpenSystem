/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
#ifndef API_UNITS_TIME_DELTA_H_
#define API_UNITS_TIME_DELTA_H_

#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>

#include "rtc_base/system/rtc_export.h"
#include "rtc_base/units/unit_base.h"  // IWYU pragma: export

namespace webrtc {

// TimeDelta represents the difference between two timestamps. Commonly this can
// be a duration. However since two Timestamps are not guaranteed to have the
// same epoch (they might come from different computers, making exact
// synchronisation infeasible), the duration covered by a TimeDelta can be
// undefined. To simplify usage, it can be constructed and converted to
// different units, specifically seconds (s), milliseconds (ms) and
// microseconds (us).
class TimeDelta final : public rtc_units_impl::RelativeUnit<TimeDelta> {
 public:
  template <typename T>
  static constexpr TimeDelta Minutes(T value) {
    static_assert(std::is_arithmetic<T>::value, "");
    return Seconds(value * 60);
  }
  template <typename T>
  static constexpr TimeDelta Seconds(T value) {
    static_assert(std::is_arithmetic<T>::value, "");
    return FromFraction(1'000'000, value);
  }
  template <typename T>
  static constexpr TimeDelta Millis(T value) {
    static_assert(std::is_arithmetic<T>::value, "");
    return FromFraction(1'000, value);
  }
  template <typename T>
  static constexpr TimeDelta Micros(T value) {
    static_assert(std::is_arithmetic<T>::value, "");
    return FromValue(value);
  }

  TimeDelta() = delete;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, TimeDelta value);

  template <typename T = int64_t>
  constexpr T seconds() const {
    return ToFraction<1000000, T>();
  }
  template <typename T = int64_t>
  constexpr T ms() const {
    return ToFraction<1000, T>();
  }
  template <typename T = int64_t>
  constexpr T us() const {
    return ToValue<T>();
  }
  template <typename T = int64_t>
  constexpr T ns() const {
    return ToMultiple<1000, T>();
  }

  constexpr int64_t seconds_or(int64_t fallback_value) const {
    return ToFractionOr<1000000>(fallback_value);
  }
  constexpr int64_t ms_or(int64_t fallback_value) const {
    return ToFractionOr<1000>(fallback_value);
  }
  constexpr int64_t us_or(int64_t fallback_value) const {
    return ToValueOr(fallback_value);
  }

  constexpr TimeDelta Abs() const {
    return us() < 0 ? TimeDelta::Micros(-us()) : *this;
  }

 private:
  friend class rtc_units_impl::UnitBase<TimeDelta>;
  using RelativeUnit::RelativeUnit;
  static constexpr bool one_sided = false;
};

RTC_EXPORT std::string ToString(TimeDelta value);

template <typename Sink>
void AbslStringify(Sink& sink, TimeDelta value) {
  sink.Append(ToString(value));
}

}  // namespace webrtc

#endif  // API_UNITS_TIME_DELTA_H_
