/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_GENERIC_PACKET_RECEIVED_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_GENERIC_PACKET_RECEIVED_H_

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

struct LoggedGenericPacketReceived {
  LoggedGenericPacketReceived() = default;
  LoggedGenericPacketReceived(Timestamp timestamp,
                              int64_t packet_number,
                              int packet_length)
      : timestamp(timestamp),
        packet_number(packet_number),
        packet_length(packet_length) {}

  int64_t log_time_us() const { return timestamp.us(); }
  int64_t log_time_ms() const { return timestamp.ms(); }
  Timestamp log_time() const { return timestamp; }

  Timestamp timestamp = Timestamp::MinusInfinity();
  int64_t packet_number;
  int packet_length;
};

class RtcEventGenericPacketReceived final : public RtcEvent {
 public:
  static constexpr Type kType = Type::GenericPacketReceived;

  RtcEventGenericPacketReceived(int64_t packet_number, size_t packet_length);
  ~RtcEventGenericPacketReceived() override;

  std::unique_ptr<RtcEventGenericPacketReceived> Copy() const;

  Type GetType() const override { return kType; }
  bool IsConfigEvent() const override { return false; }

  // An identifier of the packet.
  int64_t packet_number() const { return packet_number_; }

  // Total packet length, including all packetization overheads, but not
  // including ICE/TURN/IP overheads.
  size_t packet_length() const { return packet_length_; }

  static std::string Encode(rtc::ArrayView<const RtcEvent*> /* batch */) {
    // TODO(terelius): Implement
    return "";
  }

  static RtcEventLogParseStatus Parse(
      absl::string_view /* encoded_bytes */,
      bool /* batched */,
      std::vector<LoggedGenericPacketReceived>& /* output */) {
    // TODO(terelius): Implement
    return RtcEventLogParseStatus::Error("Not Implemented", __FILE__, __LINE__);
  }

 private:
  RtcEventGenericPacketReceived(const RtcEventGenericPacketReceived& packet);

  const int64_t packet_number_;
  const size_t packet_length_;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_GENERIC_PACKET_RECEIVED_H_
