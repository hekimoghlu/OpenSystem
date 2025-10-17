/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#include "modules/rtp_rtcp/source/rtcp_packet.h"

#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace rtcp {

rtc::Buffer RtcpPacket::Build() const {
  rtc::Buffer packet(BlockLength());

  size_t length = 0;
  bool created = Create(packet.data(), &length, packet.capacity(), nullptr);
  RTC_DCHECK(created) << "Invalid packet is not supported.";
  RTC_DCHECK_EQ(length, packet.size())
      << "BlockLength mispredicted size used by Create";

  return packet;
}

bool RtcpPacket::Build(size_t max_length, PacketReadyCallback callback) const {
  RTC_CHECK_LE(max_length, IP_PACKET_SIZE);
  uint8_t buffer[IP_PACKET_SIZE];
  size_t index = 0;
  if (!Create(buffer, &index, max_length, callback))
    return false;
  return OnBufferFull(buffer, &index, callback);
}

bool RtcpPacket::OnBufferFull(uint8_t* packet,
                              size_t* index,
                              PacketReadyCallback callback) const {
  if (*index == 0)
    return false;
  RTC_DCHECK(callback) << "Fragmentation not supported.";
  callback(rtc::ArrayView<const uint8_t>(packet, *index));
  *index = 0;
  return true;
}

size_t RtcpPacket::HeaderLength() const {
  size_t length_in_bytes = BlockLength();
  RTC_DCHECK_GT(length_in_bytes, 0);
  RTC_DCHECK_EQ(length_in_bytes % 4, 0)
      << "Padding must be handled by each subclass.";
  // Length in 32-bit words without common header.
  return (length_in_bytes - kHeaderLength) / 4;
}

// From RFC 3550, RTP: A Transport Protocol for Real-Time Applications.
//
// RTP header format.
//   0                   1                   2                   3
//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |V=2|P| RC/FMT  |      PT       |             length            |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
void RtcpPacket::CreateHeader(
    size_t count_or_format,  // Depends on packet type.
    uint8_t packet_type,
    size_t length,
    uint8_t* buffer,
    size_t* pos) {
  CreateHeader(count_or_format, packet_type, length, /*padding=*/false, buffer,
               pos);
}

void RtcpPacket::CreateHeader(
    size_t count_or_format,  // Depends on packet type.
    uint8_t packet_type,
    size_t length,
    bool padding,
    uint8_t* buffer,
    size_t* pos) {
  RTC_DCHECK_LE(length, 0xffffU);
  RTC_DCHECK_LE(count_or_format, 0x1f);
  constexpr uint8_t kVersionBits = 2 << 6;
  uint8_t padding_bit = padding ? 1 << 5 : 0;
  buffer[*pos + 0] =
      kVersionBits | padding_bit | static_cast<uint8_t>(count_or_format);
  buffer[*pos + 1] = packet_type;
  buffer[*pos + 2] = (length >> 8) & 0xff;
  buffer[*pos + 3] = length & 0xff;
  *pos += kHeaderLength;
}

}  // namespace rtcp
}  // namespace webrtc
