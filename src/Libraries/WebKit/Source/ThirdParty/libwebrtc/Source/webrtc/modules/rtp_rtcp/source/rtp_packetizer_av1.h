/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_PACKETIZER_AV1_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_PACKETIZER_AV1_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "api/array_view.h"
#include "api/video/video_frame_type.h"
#include "modules/rtp_rtcp/source/rtp_format.h"

namespace webrtc {

class RtpPacketizerAv1 : public RtpPacketizer {
 public:
  RtpPacketizerAv1(rtc::ArrayView<const uint8_t> payload,
                   PayloadSizeLimits limits,
                   VideoFrameType frame_type,
                   bool is_last_frame_in_picture,
                   bool even_distribution);
  ~RtpPacketizerAv1() override = default;

  size_t NumPackets() const override { return packets_.size() - packet_index_; }
  bool NextPacket(RtpPacketToSend* packet) override;

 private:
  struct Obu {
    uint8_t header;
    uint8_t extension_header;  // undefined if (header & kXbit) == 0
    rtc::ArrayView<const uint8_t> payload;
    int size;  // size of the header and payload combined.
  };
  struct Packet {
    explicit Packet(int first_obu_index) : first_obu(first_obu_index) {}
    // Indexes into obus_ vector of the first and last obus that should put into
    // the packet.
    int first_obu;
    int num_obu_elements = 0;
    int first_obu_offset = 0;
    int last_obu_size;
    // Total size consumed by the packet.
    int packet_size = 0;
  };

  // Parses the payload into serie of OBUs.
  static std::vector<Obu> ParseObus(rtc::ArrayView<const uint8_t> payload);
  // Returns the number of additional bytes needed to store the previous OBU
  // element if an additonal OBU element is added to the packet.
  static int AdditionalBytesForPreviousObuElement(const Packet& packet);
  // Packetize and try to distribute the payload evenly across packets.
  static std::vector<Packet> PacketizeAboutEqually(
      rtc::ArrayView<const Obu> obus,
      PayloadSizeLimits limits);
  static std::vector<Packet> Packetize(rtc::ArrayView<const Obu> obus,
                                       PayloadSizeLimits limits);

  uint8_t AggregationHeader() const;

  const VideoFrameType frame_type_;
  const std::vector<Obu> obus_;
  const std::vector<Packet> packets_;
  const bool is_last_frame_in_picture_;
  size_t packet_index_ = 0;
};

}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTP_PACKETIZER_AV1_H_
