/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
#ifndef API_UNITS_FREQUENCY_H_
#define API_UNITS_FREQUENCY_H_

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>

#include "api/units/time_delta.h"
#include "rtc_base/checks.h"
#include "rtc_base/system/rtc_export.h"
#include "rtc_base/units/unit_base.h"  // IWYU pragma: export

namespace webrtc {

class Frequency final : public rtc_units_impl::RelativeUnit<Frequency> {
 public:
  template <typename T>
  static constexpr Frequency MilliHertz(T value) {
    static_assert(std::is_arithmetic<T>::value, "");
    return FromValue(value);
  }
  template <typename T>
  static constexpr Frequency Hertz(T value) {
    static_assert(std::is_arithmetic<T>::value, "");
    return FromFraction(1'000, value);
  }
  template <typename T>
  static constexpr Frequency KiloHertz(T value) {
    static_assert(std::is_arithmetic<T>::value, "");
    return FromFraction(1'000'000, value);
  }

  Frequency() = delete;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, Frequency value);

  template <typename T = int64_t>
  constexpr T hertz() const {
    return ToFraction<1000, T>();
  }
  template <typename T = int64_t>
  constexpr T millihertz() const {
    return ToValue<T>();
  }

 private:
  friend class rtc_units_impl::UnitBase<Frequency>;
  using RelativeUnit::RelativeUnit;
  static constexpr bool one_sided = true;
};

inline constexpr Frequency operator/(int64_t nominator,
                                     const TimeDelta& interval) {
  constexpr int64_t kKiloPerMicro = 1000 * 1000000;
  RTC_DCHECK_LE(nominator, std::numeric_limits<int64_t>::max() / kKiloPerMicro);
  RTC_CHECK(interval.IsFinite());
  RTC_CHECK(!interval.IsZero());
  return Frequency::MilliHertz(nominator * kKiloPerMicro / interval.us());
}

inline constexpr TimeDelta operator/(int64_t nominator,
                                     const Frequency& frequency) {
  constexpr int64_t kMegaPerMilli = 1000000 * 1000;
  RTC_DCHECK_LE(nominator, std::numeric_limits<int64_t>::max() / kMegaPerMilli);
  RTC_CHECK(frequency.IsFinite());
  RTC_CHECK(!frequency.IsZero());
  return TimeDelta::Micros(nominator * kMegaPerMilli / frequency.millihertz());
}

inline constexpr double operator*(Frequency frequency, TimeDelta time_delta) {
  return frequency.hertz<double>() * time_delta.seconds<double>();
}
inline constexpr double operator*(TimeDelta time_delta, Frequency frequency) {
  return frequency * time_delta;
}

RTC_EXPORT std::string ToString(Frequency value);

template <typename Sink>
void AbslStringify(Sink& sink, Frequency value) {
  sink.Append(ToString(value));
}

}  // namespace webrtc
#endif  // API_UNITS_FREQUENCY_H_
