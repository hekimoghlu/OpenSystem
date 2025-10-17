/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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
#ifndef MODULES_RTP_RTCP_SOURCE_FEC_TEST_HELPER_H_
#define MODULES_RTP_RTCP_SOURCE_FEC_TEST_HELPER_H_

#include <memory>

#include "modules/rtp_rtcp/source/forward_error_correction.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"
#include "rtc_base/random.h"

namespace webrtc {
namespace test {
namespace fec {

struct AugmentedPacket : public ForwardErrorCorrection::Packet {
  RTPHeader header;
};

// TODO(brandtr): Consider merging MediaPacketGenerator and
// AugmentedPacketGenerator into a single class, since their functionality is
// similar.

// This class generates media packets corresponding to a single frame.
class MediaPacketGenerator {
 public:
  MediaPacketGenerator(uint32_t min_packet_size,
                       uint32_t max_packet_size,
                       uint32_t ssrc,
                       Random* random);
  ~MediaPacketGenerator();

  // Construct the media packets, up to `num_media_packets` packets.
  ForwardErrorCorrection::PacketList ConstructMediaPackets(
      int num_media_packets,
      uint16_t start_seq_num);
  ForwardErrorCorrection::PacketList ConstructMediaPackets(
      int num_media_packets);

  uint16_t GetNextSeqNum();

 private:
  uint32_t min_packet_size_;
  uint32_t max_packet_size_;
  uint32_t ssrc_;
  Random* random_;

  ForwardErrorCorrection::PacketList media_packets_;
  uint16_t next_seq_num_;
};

// This class generates media packets with a certain structure of the payload.
class AugmentedPacketGenerator {
 public:
  explicit AugmentedPacketGenerator(uint32_t ssrc);

  // Prepare for generating a new set of packets, corresponding to a frame.
  void NewFrame(size_t num_packets);

  // Increment and return the newly incremented sequence number.
  uint16_t NextPacketSeqNum();

  // Return the next packet in the current frame.
  std::unique_ptr<AugmentedPacket> NextPacket(size_t offset, size_t length);

 protected:
  // Given `header`, writes the appropriate RTP header fields in `data`.
  static void WriteRtpHeader(const RTPHeader& header, uint8_t* data);

  // Number of packets left to generate, in the current frame.
  size_t num_packets_;

 private:
  uint32_t ssrc_;
  uint16_t seq_num_;
  uint32_t timestamp_;
};

// This class generates media and FlexFEC packets for a single frame.
class FlexfecPacketGenerator : public AugmentedPacketGenerator {
 public:
  FlexfecPacketGenerator(uint32_t media_ssrc, uint32_t flexfec_ssrc);

  // Creates a new AugmentedPacket (with RTP headers) from a
  // FlexFEC packet (without RTP headers).
  std::unique_ptr<AugmentedPacket> BuildFlexfecPacket(
      const ForwardErrorCorrection::Packet& packet);

 private:
  uint32_t flexfec_ssrc_;
  uint16_t flexfec_seq_num_;
  uint32_t flexfec_timestamp_;
};

// This class generates media and ULPFEC packets (both encapsulated in RED)
// for a single frame.
class UlpfecPacketGenerator : public AugmentedPacketGenerator {
 public:
  explicit UlpfecPacketGenerator(uint32_t ssrc);

  // Creates a new RtpPacket with the RED header added to the packet.
  static RtpPacketReceived BuildMediaRedPacket(const AugmentedPacket& packet,
                                               bool is_recovered);

  // Creates a new RtpPacket with FEC payload and RED header. Does this by
  // creating a new fake media AugmentedPacket, clears the marker bit and adds a
  // RED header. Finally replaces the payload with the content of
  // `packet->data`.
  RtpPacketReceived BuildUlpfecRedPacket(
      const ForwardErrorCorrection::Packet& packet);
};

}  // namespace fec
}  // namespace test
}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_FEC_TEST_HELPER_H_
