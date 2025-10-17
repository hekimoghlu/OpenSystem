/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
#ifndef MODULES_RTP_RTCP_SOURCE_PACKET_SEQUENCER_H_
#define MODULES_RTP_RTCP_SOURCE_PACKET_SEQUENCER_H_

#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/rtp_packet_to_send.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

// Helper class used to assign RTP sequence numbers and populate some fields for
// padding packets based on the last sequenced packets.
// This class is not thread safe, the caller must provide that.
class PacketSequencer {
 public:
  // If `require_marker_before_media_padding_` is true, padding packets on the
  // media ssrc is not allowed unless the last sequenced media packet had the
  // marker bit set (i.e. don't insert padding packets between the first and
  // last packets of a video frame).
  // Packets with unknown SSRCs will be ignored.
  PacketSequencer(uint32_t media_ssrc,
                  std::optional<uint32_t> rtx_ssrc,
                  bool require_marker_before_media_padding,
                  Clock* clock);

  // Assigns sequence number, and in the case of non-RTX padding also timestamps
  // and payload type.
  void Sequence(RtpPacketToSend& packet);

  void set_media_sequence_number(uint16_t sequence_number) {
    media_sequence_number_ = sequence_number;
  }
  void set_rtx_sequence_number(uint16_t sequence_number) {
    rtx_sequence_number_ = sequence_number;
  }

  void SetRtpState(const RtpState& state);
  void PopulateRtpState(RtpState& state) const;

  uint16_t media_sequence_number() const { return media_sequence_number_; }
  uint16_t rtx_sequence_number() const { return rtx_sequence_number_; }

  // Checks whether it is allowed to send padding on the media SSRC at this
  // time, e.g. that we don't send padding in the middle of a video frame.
  bool CanSendPaddingOnMediaSsrc() const;

 private:
  void UpdateLastPacketState(const RtpPacketToSend& packet);
  void PopulatePaddingFields(RtpPacketToSend& packet);

  const uint32_t media_ssrc_;
  const std::optional<uint32_t> rtx_ssrc_;
  const bool require_marker_before_media_padding_;
  Clock* const clock_;

  uint16_t media_sequence_number_;
  uint16_t rtx_sequence_number_;

  int8_t last_payload_type_;
  uint32_t last_rtp_timestamp_;
  Timestamp last_capture_time_ = Timestamp::MinusInfinity();
  Timestamp last_timestamp_time_ = Timestamp::MinusInfinity();
  bool last_packet_marker_bit_;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_PACKET_SEQUENCER_H_
