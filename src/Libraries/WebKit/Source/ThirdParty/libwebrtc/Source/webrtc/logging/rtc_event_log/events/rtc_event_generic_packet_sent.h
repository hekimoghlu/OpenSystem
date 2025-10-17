/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_GENERIC_PACKET_SENT_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_GENERIC_PACKET_SENT_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/units/timestamp.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"

namespace webrtc {

struct LoggedGenericPacketSent {
  LoggedGenericPacketSent() = default;
  LoggedGenericPacketSent(Timestamp timestamp,
                          int64_t packet_number,
                          size_t overhead_length,
                          size_t payload_length,
                          size_t padding_length)
      : timestamp(timestamp),
        packet_number(packet_number),
        overhead_length(overhead_length),
        payload_length(payload_length),
        padding_length(padding_length) {}

  int64_t log_time_us() const { return timestamp.us(); }
  int64_t log_time_ms() const { return timestamp.ms(); }
  Timestamp log_time() const { return timestamp; }

  size_t packet_length() const {
    return payload_length + padding_length + overhead_length;
  }
  Timestamp timestamp = Timestamp::MinusInfinity();
  int64_t packet_number;
  size_t overhead_length;
  size_t payload_length;
  size_t padding_length;
};

class RtcEventGenericPacketSent final : public RtcEvent {
 public:
  static constexpr Type kType = Type::GenericPacketSent;

  RtcEventGenericPacketSent(int64_t packet_number,
                            size_t overhead_length,
                            size_t payload_length,
                            size_t padding_length);
  ~RtcEventGenericPacketSent() override;

  std::unique_ptr<RtcEventGenericPacketSent> Copy() const;

  Type GetType() const override { return kType; }
  bool IsConfigEvent() const override { return false; }

  // An identifier of the packet.
  int64_t packet_number() const { return packet_number_; }

  // Total packet length, including all packetization overheads, but not
  // including ICE/TURN/IP overheads.
  size_t packet_length() const {
    return overhead_length_ + payload_length_ + padding_length_;
  }

  // The number of bytes in overhead, including framing overheads, acks if
  // present, etc.
  size_t overhead_length() const { return overhead_length_; }

  // Length of payload sent (size of raw audio/video/data), without
  // packetization overheads. This may still include serialization overheads.
  size_t payload_length() const { return payload_length_; }

  size_t padding_length() const { return padding_length_; }

  static std::string Encode(rtc::ArrayView<const RtcEvent*> /* batch */) {
    // TODO(terelius): Implement
    return "";
  }

  static RtcEventLogParseStatus Parse(
      absl::string_view /* encoded_bytes */,
      bool /* batched */,
      std::vector<LoggedGenericPacketSent>& /* output */) {
    // TODO(terelius): Implement
    return RtcEventLogParseStatus::Error("Not Implemented", __FILE__, __LINE__);
  }

 private:
  RtcEventGenericPacketSent(const RtcEventGenericPacketSent& packet);

  const int64_t packet_number_;
  const size_t overhead_length_;
  const size_t payload_length_;
  const size_t padding_length_;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_GENERIC_PACKET_SENT_H_
