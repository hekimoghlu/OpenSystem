/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_RTPFB_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_RTPFB_H_

#include <stddef.h>
#include <stdint.h>

#include "modules/rtp_rtcp/source/rtcp_packet.h"

namespace webrtc {
namespace rtcp {

// RTPFB: Transport layer feedback message.
// RFC4585, Section 6.2
class Rtpfb : public RtcpPacket {
 public:
  static constexpr uint8_t kPacketType = 205;

  Rtpfb() = default;
  ~Rtpfb() override = default;

  void SetMediaSsrc(uint32_t ssrc) { media_ssrc_ = ssrc; }

  uint32_t media_ssrc() const { return media_ssrc_; }

 protected:
  static constexpr size_t kCommonFeedbackLength = 8;
  void ParseCommonFeedback(const uint8_t* payload);
  void CreateCommonFeedback(uint8_t* payload) const;

 private:
  uint32_t media_ssrc_ = 0;
};

}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_RTPFB_H_
