/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_FORMAT_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_FORMAT_H_

#include <stdint.h>

#include <memory>
#include <optional>
#include <vector>

#include "api/array_view.h"
#include "modules/rtp_rtcp/source/rtp_video_header.h"

namespace webrtc {

class RtpPacketToSend;

class RtpPacketizer {
 public:
  struct PayloadSizeLimits {
    int max_payload_len = 1200;
    int first_packet_reduction_len = 0;
    int last_packet_reduction_len = 0;
    // Reduction len for packet that is first & last at the same time.
    int single_packet_reduction_len = 0;
  };

  // If type is not set, returns a raw packetizer.
  static std::unique_ptr<RtpPacketizer> Create(
      std::optional<VideoCodecType> type,
      rtc::ArrayView<const uint8_t> payload,
      PayloadSizeLimits limits,
      // Codec-specific details.
      const RTPVideoHeader& rtp_video_header,
      // TODO(bugs.webrtc.org/15927): remove after rollout.
      bool enable_av1_even_split = false);

  virtual ~RtpPacketizer() = default;

  // Returns number of remaining packets to produce by the packetizer.
  virtual size_t NumPackets() const = 0;

  // Get the next payload with payload header.
  // Write payload and set marker bit of the `packet`.
  // Returns true on success, false otherwise.
  virtual bool NextPacket(RtpPacketToSend* packet) = 0;

  // Split payload_len into sum of integers with respect to `limits`.
  // Returns empty vector on failure.
  static std::vector<int> SplitAboutEqually(int payload_len,
                                            const PayloadSizeLimits& limits);
};
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTP_FORMAT_H_
