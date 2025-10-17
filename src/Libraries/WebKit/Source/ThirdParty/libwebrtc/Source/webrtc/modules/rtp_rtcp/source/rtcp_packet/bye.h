/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_BYE_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_BYE_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "modules/rtp_rtcp/source/rtcp_packet.h"

namespace webrtc {
namespace rtcp {
class CommonHeader;

class Bye : public RtcpPacket {
 public:
  static constexpr uint8_t kPacketType = 203;

  Bye();
  ~Bye() override;

  // Parse assumes header is already parsed and validated.
  bool Parse(const CommonHeader& packet);

  bool SetCsrcs(std::vector<uint32_t> csrcs);
  void SetReason(absl::string_view reason);

  const std::vector<uint32_t>& csrcs() const { return csrcs_; }
  const std::string& reason() const { return reason_; }

  size_t BlockLength() const override;

  bool Create(uint8_t* packet,
              size_t* index,
              size_t max_length,
              PacketReadyCallback callback) const override;

 private:
  static constexpr int kMaxNumberOfCsrcs =
      0x1f - 1;  // First item is sender SSRC.

  std::vector<uint32_t> csrcs_;
  std::string reason_;
};

}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_BYE_H_
