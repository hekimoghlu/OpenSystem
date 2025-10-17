/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_RAPID_RESYNC_REQUEST_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_RAPID_RESYNC_REQUEST_H_

#include "modules/rtp_rtcp/source/rtcp_packet/rtpfb.h"

namespace webrtc {
namespace rtcp {
class CommonHeader;

// draft-perkins-avt-rapid-rtp-sync-03
class RapidResyncRequest : public Rtpfb {
 public:
  static constexpr uint8_t kFeedbackMessageType = 5;

  RapidResyncRequest() {}
  ~RapidResyncRequest() override {}

  // Parse assumes header is already parsed and validated.
  bool Parse(const CommonHeader& header);

  size_t BlockLength() const override;

  bool Create(uint8_t* packet,
              size_t* index,
              size_t max_length,
              PacketReadyCallback callback) const override;
};
}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_RAPID_RESYNC_REQUEST_H_
