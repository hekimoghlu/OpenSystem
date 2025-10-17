/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#ifndef API_UNITS_DATA_SIZE_H_
#define API_UNITS_DATA_SIZE_H_

#include <cstdint>
#include <string>
#include <type_traits>

#include "rtc_base/system/rtc_export.h"
#include "rtc_base/units/unit_base.h"  // IWYU pragma: export

namespace webrtc {
// DataSize is a class represeting a count of bytes.
class DataSize final : public rtc_units_impl::RelativeUnit<DataSize> {
 public:
  template <typename T>
  static constexpr DataSize Bytes(T value) {
    static_assert(std::is_arithmetic<T>::value, "");
    return FromValue(value);
  }
  static constexpr DataSize Infinity() { return PlusInfinity(); }

  DataSize() = delete;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, DataSize value);

  template <typename T = int64_t>
  constexpr T bytes() const {
    return ToValue<T>();
  }

  constexpr int64_t bytes_or(int64_t fallback_value) const {
    return ToValueOr(fallback_value);
  }

 private:
  friend class rtc_units_impl::UnitBase<DataSize>;
  using RelativeUnit::RelativeUnit;
  static constexpr bool one_sided = true;
};

RTC_EXPORT std::string ToString(DataSize value);

template <typename Sink>
void AbslStringify(Sink& sink, DataSize value) {
  sink.Append(ToString(value));
}

}  // namespace webrtc

#endif  // API_UNITS_DATA_SIZE_H_
