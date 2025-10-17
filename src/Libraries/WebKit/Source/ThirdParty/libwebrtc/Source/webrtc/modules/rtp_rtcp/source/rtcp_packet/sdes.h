/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_SDES_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_SDES_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "modules/rtp_rtcp/source/rtcp_packet.h"

namespace webrtc {
namespace rtcp {
class CommonHeader;
// Source Description (SDES) (RFC 3550).
class Sdes : public RtcpPacket {
 public:
  struct Chunk {
    uint32_t ssrc;
    std::string cname;
  };
  static constexpr uint8_t kPacketType = 202;
  static constexpr size_t kMaxNumberOfChunks = 0x1f;

  Sdes();
  ~Sdes() override;

  // Parse assumes header is already parsed and validated.
  bool Parse(const CommonHeader& packet);

  bool AddCName(uint32_t ssrc, absl::string_view cname);

  const std::vector<Chunk>& chunks() const { return chunks_; }

  size_t BlockLength() const override;

  bool Create(uint8_t* packet,
              size_t* index,
              size_t max_length,
              PacketReadyCallback callback) const override;

 private:
  std::vector<Chunk> chunks_;
  size_t block_length_;
};
}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_SDES_H_
