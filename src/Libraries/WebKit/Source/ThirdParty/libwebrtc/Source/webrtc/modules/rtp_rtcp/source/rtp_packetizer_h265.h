/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_PACKETIZER_H265_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_PACKETIZER_H265_H_

#include <deque>
#include <queue>
#include <string>

#include "api/array_view.h"
#include "modules/rtp_rtcp/source/rtp_format.h"
#include "modules/rtp_rtcp/source/rtp_packet_to_send.h"

namespace webrtc {

class RtpPacketizerH265 : public RtpPacketizer {
 public:
  // Initialize with payload from encoder.
  // The payload_data must be exactly one encoded H.265 frame.
  // For H265 we only support tx-mode SRST.
  RtpPacketizerH265(rtc::ArrayView<const uint8_t> payload,
                    PayloadSizeLimits limits);

  RtpPacketizerH265(const RtpPacketizerH265&) = delete;
  RtpPacketizerH265& operator=(const RtpPacketizerH265&) = delete;

  ~RtpPacketizerH265() override;

  size_t NumPackets() const override;

  // Get the next payload with H.265 payload header.
  // Write payload and set marker bit of the `packet`.
  // Returns true on success or false if there was no payload to packetize.
  bool NextPacket(RtpPacketToSend* rtp_packet) override;

 private:
  struct PacketUnit {
    rtc::ArrayView<const uint8_t> source_fragment;
    bool first_fragment = false;
    bool last_fragment = false;
    bool aggregated = false;
    uint16_t header = 0;
  };
  std::deque<rtc::ArrayView<const uint8_t>> input_fragments_;
  std::queue<PacketUnit> packets_;

  bool GeneratePackets();
  bool PacketizeFu(size_t fragment_index);
  int PacketizeAp(size_t fragment_index);

#if WEBRTC_WEBKIT_BUILD
  bool NextAggregatePacket(RtpPacketToSend* rtp_packet);
#else
  void NextAggregatePacket(RtpPacketToSend* rtp_packet);
#endif
  void NextFragmentPacket(RtpPacketToSend* rtp_packet);

  const PayloadSizeLimits limits_;
  size_t num_packets_left_ = 0;
};
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTP_PACKETIZER_H265_H_
