/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
#ifndef RTC_BASE_NUMERICS_SEQUENCE_NUMBER_UNWRAPPER_H_
#define RTC_BASE_NUMERICS_SEQUENCE_NUMBER_UNWRAPPER_H_

#include <stdint.h>

#include <limits>
#include <optional>
#include <type_traits>

#include "rtc_base/numerics/sequence_number_util.h"

namespace webrtc {

// A sequence number unwrapper where the first unwrapped value equals the
// first value being unwrapped.
template <typename T, T M = 0>
class SeqNumUnwrapper {
  static_assert(
      std::is_unsigned<T>::value &&
          std::numeric_limits<T>::max() < std::numeric_limits<int64_t>::max(),
      "Type unwrapped must be an unsigned integer smaller than int64_t.");

 public:
  // Unwraps `value` and updates the internal state of the unwrapper.
  int64_t Unwrap(T value) {
    if (!last_value_) {
      last_unwrapped_ = {value};
    } else {
      last_unwrapped_ += Delta(*last_value_, value);
    }

    last_value_ = value;
    return last_unwrapped_;
  }

  // Returns the `value` without updating the internal state of the unwrapper.
  int64_t PeekUnwrap(T value) const {
    if (!last_value_) {
      return value;
    }
    return last_unwrapped_ + Delta(*last_value_, value);
  }

  // Resets the unwrapper to its initial state. Unwrapped sequence numbers will
  // being at 0 after resetting.
  void Reset() {
    last_unwrapped_ = 0;
    last_value_.reset();
  }

 private:
  static int64_t Delta(T last_value, T new_value) {
    constexpr int64_t kBackwardAdjustment =
        M == 0 ? int64_t{std::numeric_limits<T>::max()} + 1 : M;
    int64_t result = ForwardDiff<T, M>(last_value, new_value);
    if (!AheadOrAt<T, M>(new_value, last_value)) {
      result -= kBackwardAdjustment;
    }
    return result;
  }

  int64_t last_unwrapped_ = 0;
  std::optional<T> last_value_;
};

using RtpTimestampUnwrapper = SeqNumUnwrapper<uint32_t>;
using RtpSequenceNumberUnwrapper = SeqNumUnwrapper<uint16_t>;

}  // namespace webrtc

#endif  // RTC_BASE_NUMERICS_SEQUENCE_NUMBER_UNWRAPPER_H_
