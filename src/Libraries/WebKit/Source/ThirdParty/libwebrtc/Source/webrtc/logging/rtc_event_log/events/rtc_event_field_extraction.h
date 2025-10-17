/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_FIELD_EXTRACTION_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_FIELD_EXTRACTION_H_

#include <cstdint>
#include <type_traits>
#include <vector>

#include "logging/rtc_event_log/encoder/rtc_event_log_encoder_common.h"
#include "rtc_base/logging.h"

namespace webrtc_event_logging {
uint8_t UnsignedBitWidth(uint64_t max_magnitude);
uint8_t SignedBitWidth(uint64_t max_pos_magnitude, uint64_t max_neg_magnitude);
uint64_t MaxUnsignedValueOfBitWidth(uint64_t bit_width);
uint64_t UnsignedDelta(uint64_t previous, uint64_t current, uint64_t bit_mask);
}  // namespace webrtc_event_logging

namespace webrtc {
template <typename T, std::enable_if_t<std::is_signed<T>::value, bool> = true>
uint64_t EncodeAsUnsigned(T value) {
  return webrtc_event_logging::ToUnsigned(value);
}

template <typename T, std::enable_if_t<std::is_unsigned<T>::value, bool> = true>
uint64_t EncodeAsUnsigned(T value) {
  return static_cast<uint64_t>(value);
}

template <typename T, std::enable_if_t<std::is_signed<T>::value, bool> = true>
T DecodeFromUnsignedToType(uint64_t value) {
  T signed_value = 0;
  bool success = webrtc_event_logging::ToSigned<T>(value, &signed_value);
  if (!success) {
    RTC_LOG(LS_ERROR) << "Failed to convert " << value << "to signed type.";
    // TODO(terelius): Propagate error?
  }
  return signed_value;
}

template <typename T, std::enable_if_t<std::is_unsigned<T>::value, bool> = true>
T DecodeFromUnsignedToType(uint64_t value) {
  // TODO(terelius): Check range?
  return static_cast<T>(value);
}

// RtcEventLogEnum<T> defines a mapping between an enum T
// and the event log encodings. To log a new enum type T,
// specialize RtcEventLogEnum<T> and add static methods
// static uint64_t Encode(T x) {}
// static RtcEventLogParseStatusOr<T> Decode(uint64_t x) {}
template <typename T>
class RtcEventLogEnum {
  static_assert(sizeof(T) != sizeof(T),
                "Missing specialisation of RtcEventLogEnum for type");
};

// Represents a vector<optional<uint64_t>> optional_values
// as a bit-vector `position_mask` which identifies the positions
// of existing values, and a (potentially shorter)
// `vector<uint64_t> values` containing the actual values.
// The bit vector is constructed such that position_mask[i]
// is true iff optional_values[i] has a value, and `values.size()`
// is equal to the number of set bits in `position_mask`.
struct ValuesWithPositions {
  std::vector<bool> position_mask;
  std::vector<uint64_t> values;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_FIELD_EXTRACTION_H_
