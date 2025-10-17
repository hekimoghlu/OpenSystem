/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
#include <cstdint>
#include <optional>

#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/rtp_header_extensions.h"
#include "modules/rtp_rtcp/source/rtp_packet_to_send.h"

namespace webrtc {

RtpPacketSendInfo RtpPacketSendInfo::From(const RtpPacketToSend& packet,
                                          const PacedPacketInfo& pacing_info) {
  RtpPacketSendInfo packet_info;
  if (packet.transport_sequence_number()) {
    packet_info.transport_sequence_number =
        *packet.transport_sequence_number() & 0xFFFF;
  } else {
    std::optional<uint16_t> packet_id =
        packet.GetExtension<TransportSequenceNumber>();
    if (packet_id) {
      packet_info.transport_sequence_number = *packet_id;
    }
  }

  packet_info.rtp_timestamp = packet.Timestamp();
  packet_info.length = packet.size();
  packet_info.pacing_info = pacing_info;
  packet_info.packet_type = packet.packet_type();

  switch (*packet_info.packet_type) {
    case RtpPacketMediaType::kAudio:
    case RtpPacketMediaType::kVideo:
      packet_info.media_ssrc = packet.Ssrc();
      packet_info.rtp_sequence_number = packet.SequenceNumber();
      break;
    case RtpPacketMediaType::kRetransmission:
      RTC_DCHECK(packet.original_ssrc() &&
                 packet.retransmitted_sequence_number());
      // For retransmissions, we're want to remove the original media packet
      // if the retransmit arrives - so populate that in the packet info.
      packet_info.media_ssrc = packet.original_ssrc().value_or(0);
      packet_info.rtp_sequence_number =
          packet.retransmitted_sequence_number().value_or(0);
      break;
    case RtpPacketMediaType::kPadding:
    case RtpPacketMediaType::kForwardErrorCorrection:
      // We're not interested in feedback about these packets being received
      // or lost.
      break;
  }
  return packet_info;
}

}  // namespace webrtc
