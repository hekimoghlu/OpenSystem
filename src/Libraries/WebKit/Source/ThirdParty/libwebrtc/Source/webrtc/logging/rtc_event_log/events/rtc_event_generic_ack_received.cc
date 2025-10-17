/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
#include "logging/rtc_event_log/events/rtc_event_generic_ack_received.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/memory/memory.h"
#include "api/rtc_event_log/rtc_event.h"
#include "rtc_base/time_utils.h"

namespace webrtc {

std::vector<std::unique_ptr<RtcEventGenericAckReceived>>
RtcEventGenericAckReceived::CreateLogs(
    int64_t packet_number,
    const std::vector<AckedPacket>& acked_packets) {
  std::vector<std::unique_ptr<RtcEventGenericAckReceived>> result;
  int64_t time_us = rtc::TimeMicros();
  result.reserve(acked_packets.size());
  for (const AckedPacket& packet : acked_packets) {
    result.emplace_back(new RtcEventGenericAckReceived(
        time_us, packet_number, packet.packet_number,
        packet.receive_acked_packet_time_ms));
  }
  return result;
}

RtcEventGenericAckReceived::RtcEventGenericAckReceived(
    int64_t timestamp_us,
    int64_t packet_number,
    int64_t acked_packet_number,
    std::optional<int64_t> receive_acked_packet_time_ms)
    : RtcEvent(timestamp_us),
      packet_number_(packet_number),
      acked_packet_number_(acked_packet_number),
      receive_acked_packet_time_ms_(receive_acked_packet_time_ms) {}

std::unique_ptr<RtcEventGenericAckReceived> RtcEventGenericAckReceived::Copy()
    const {
  return absl::WrapUnique(new RtcEventGenericAckReceived(*this));
}

RtcEventGenericAckReceived::RtcEventGenericAckReceived(
    const RtcEventGenericAckReceived& packet) = default;

RtcEventGenericAckReceived::~RtcEventGenericAckReceived() = default;

}  // namespace webrtc
