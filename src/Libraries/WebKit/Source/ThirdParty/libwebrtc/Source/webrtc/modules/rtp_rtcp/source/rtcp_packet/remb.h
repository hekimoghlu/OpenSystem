/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_REMB_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_REMB_H_

#include <vector>

#include "modules/rtp_rtcp/source/rtcp_packet/psfb.h"

namespace webrtc {
namespace rtcp {
class CommonHeader;

// Receiver Estimated Max Bitrate (REMB) (draft-alvestrand-rmcat-remb).
class Remb : public Psfb {
 public:
  static constexpr size_t kMaxNumberOfSsrcs = 0xff;

  Remb();
  Remb(const Remb&);
  ~Remb() override;

  // Parse assumes header is already parsed and validated.
  bool Parse(const CommonHeader& packet);

  bool SetSsrcs(std::vector<uint32_t> ssrcs);
  void SetBitrateBps(int64_t bitrate_bps) { bitrate_bps_ = bitrate_bps; }

  int64_t bitrate_bps() const { return bitrate_bps_; }
  const std::vector<uint32_t>& ssrcs() const { return ssrcs_; }

  size_t BlockLength() const override;

  bool Create(uint8_t* packet,
              size_t* index,
              size_t max_length,
              PacketReadyCallback callback) const override;

 private:
  static constexpr uint32_t kUniqueIdentifier = 0x52454D42;  // 'R' 'E' 'M' 'B'.

  // Media ssrc is unused, shadow base class setter and getter.
  void SetMediaSsrc(uint32_t);
  uint32_t media_ssrc() const;

  int64_t bitrate_bps_;
  std::vector<uint32_t> ssrcs_;
};
}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_REMB_H_
