/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_TMMB_ITEM_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_TMMB_ITEM_H_

#include <stddef.h>
#include <stdint.h>

namespace webrtc {
namespace rtcp {
// RFC5104, Section 3.5.4
// Temporary Maximum Media Stream Bitrate Request/Notification.
// Used both by TMMBR and TMMBN rtcp packets.
class TmmbItem {
 public:
  static constexpr size_t kLength = 8;

  TmmbItem() : ssrc_(0), bitrate_bps_(0), packet_overhead_(0) {}
  TmmbItem(uint32_t ssrc, uint64_t bitrate_bps, uint16_t overhead);

  bool Parse(const uint8_t* buffer);
  void Create(uint8_t* buffer) const;

  void set_ssrc(uint32_t ssrc) { ssrc_ = ssrc; }
  void set_bitrate_bps(uint64_t bitrate_bps) { bitrate_bps_ = bitrate_bps; }
  void set_packet_overhead(uint16_t overhead);

  uint32_t ssrc() const { return ssrc_; }
  uint64_t bitrate_bps() const { return bitrate_bps_; }
  uint16_t packet_overhead() const { return packet_overhead_; }

 private:
  // Media stream id.
  uint32_t ssrc_;
  // Maximum total media bit rate that the media receiver is
  // currently prepared to accept for this media stream.
  uint64_t bitrate_bps_;
  // Per-packet overhead that the media receiver has observed
  // for this media stream at its chosen reference protocol layer.
  uint16_t packet_overhead_;
};
}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_TMMB_ITEM_H_
