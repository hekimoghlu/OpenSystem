/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_FIR_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_FIR_H_

#include <vector>

#include "modules/rtp_rtcp/source/rtcp_packet/psfb.h"

namespace webrtc {
namespace rtcp {
class CommonHeader;
// Full intra request (FIR) (RFC 5104).
class Fir : public Psfb {
 public:
  static constexpr uint8_t kFeedbackMessageType = 4;
  struct Request {
    Request() : ssrc(0), seq_nr(0) {}
    Request(uint32_t ssrc, uint8_t seq_nr) : ssrc(ssrc), seq_nr(seq_nr) {}
    uint32_t ssrc;
    uint8_t seq_nr;
  };

  Fir();
  Fir(const Fir& fir);
  ~Fir() override;

  // Parse assumes header is already parsed and validated.
  bool Parse(const CommonHeader& packet);

  void AddRequestTo(uint32_t ssrc, uint8_t seq_num) {
    items_.emplace_back(ssrc, seq_num);
  }
  const std::vector<Request>& requests() const { return items_; }

  size_t BlockLength() const override;

  bool Create(uint8_t* packet,
              size_t* index,
              size_t max_length,
              PacketReadyCallback callback) const override;

 private:
  static constexpr size_t kFciLength = 8;

  // SSRC of media source is not used in FIR packet. Shadow base functions.
  void SetMediaSsrc(uint32_t ssrc);
  uint32_t media_ssrc() const;

  std::vector<Request> items_;
};
}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_FIR_H_
