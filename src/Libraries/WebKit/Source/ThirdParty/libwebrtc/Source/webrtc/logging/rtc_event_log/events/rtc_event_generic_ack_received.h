/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_GENERIC_ACK_RECEIVED_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_GENERIC_ACK_RECEIVED_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/units/timestamp.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"

namespace webrtc {

struct LoggedGenericAckReceived {
  LoggedGenericAckReceived() = default;
  LoggedGenericAckReceived(Timestamp timestamp,
                           int64_t packet_number,
                           int64_t acked_packet_number,
                           std::optional<int64_t> receive_acked_packet_time_ms)
      : timestamp(timestamp),
        packet_number(packet_number),
        acked_packet_number(acked_packet_number),
        receive_acked_packet_time_ms(receive_acked_packet_time_ms) {}

  int64_t log_time_us() const { return timestamp.us(); }
  int64_t log_time_ms() const { return timestamp.ms(); }
  Timestamp log_time() const { return timestamp; }

  Timestamp timestamp = Timestamp::MinusInfinity();
  int64_t packet_number;
  int64_t acked_packet_number;
  std::optional<int64_t> receive_acked_packet_time_ms;
};

struct AckedPacket {
  // The packet number that was acked.
  int64_t packet_number;

  // The time where the packet was received. Not every ACK will
  // include the receive timestamp.
  std::optional<int64_t> receive_acked_packet_time_ms;
};

class RtcEventGenericAckReceived final : public RtcEvent {
 public:
  static constexpr Type kType = Type::GenericAckReceived;

  // For a collection of acked packets, it creates a vector of logs to log with
  // the same timestamp.
  static std::vector<std::unique_ptr<RtcEventGenericAckReceived>> CreateLogs(
      int64_t packet_number,
      const std::vector<AckedPacket>& acked_packets);

  ~RtcEventGenericAckReceived() override;

  std::unique_ptr<RtcEventGenericAckReceived> Copy() const;

  Type GetType() const override { return kType; }
  bool IsConfigEvent() const override { return false; }

  // An identifier of the packet which contained an ack.
  int64_t packet_number() const { return packet_number_; }

  // An identifier of the acked packet.
  int64_t acked_packet_number() const { return acked_packet_number_; }

  // Timestamp when the `acked_packet_number` was received by the remote side.
  std::optional<int64_t> receive_acked_packet_time_ms() const {
    return receive_acked_packet_time_ms_;
  }

  static std::string Encode(rtc::ArrayView<const RtcEvent*> /* batch */) {
    // TODO(terelius): Implement
    return "";
  }

  static RtcEventLogParseStatus Parse(
      absl::string_view /* encoded_bytes */,
      bool /* batched */,
      std::vector<LoggedGenericAckReceived>& /* output */) {
    // TODO(terelius): Implement
    return RtcEventLogParseStatus::Error("Not Implemented", __FILE__, __LINE__);
  }

 private:
  RtcEventGenericAckReceived(const RtcEventGenericAckReceived& packet);

  // When the ack is received, `packet_number` identifies the packet which
  // contained an ack for `acked_packet_number`, and contains the
  // `receive_acked_packet_time_ms` on which the `acked_packet_number` was
  // received on the remote side. The `receive_acked_packet_time_ms` may be
  // null.
  RtcEventGenericAckReceived(
      int64_t timestamp_us,
      int64_t packet_number,
      int64_t acked_packet_number,
      std::optional<int64_t> receive_acked_packet_time_ms);

  const int64_t packet_number_;
  const int64_t acked_packet_number_;
  const std::optional<int64_t> receive_acked_packet_time_ms_;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_GENERIC_ACK_RECEIVED_H_
