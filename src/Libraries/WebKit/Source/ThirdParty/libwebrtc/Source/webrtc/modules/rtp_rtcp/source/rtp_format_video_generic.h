/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_FORMAT_VIDEO_GENERIC_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_FORMAT_VIDEO_GENERIC_H_

#include <stdint.h>

#include <vector>

#include "api/array_view.h"
#include "modules/rtp_rtcp/source/rtp_format.h"

namespace webrtc {

class RtpPacketToSend;
struct RTPVideoHeader;

namespace RtpFormatVideoGeneric {
inline constexpr uint8_t kKeyFrameBit = 0x01;
inline constexpr uint8_t kFirstPacketBit = 0x02;
// If this bit is set, there will be an extended header contained in this
// packet. This was added later so old clients will not send this.
inline constexpr uint8_t kExtendedHeaderBit = 0x04;
}  // namespace RtpFormatVideoGeneric

class RtpPacketizerGeneric : public RtpPacketizer {
 public:
  // Initialize with payload from encoder.
  // The payload_data must be exactly one encoded generic frame.
  // Packets returned by `NextPacket` will contain the generic payload header.
  RtpPacketizerGeneric(rtc::ArrayView<const uint8_t> payload,
                       PayloadSizeLimits limits,
                       const RTPVideoHeader& rtp_video_header);
  // Initialize with payload from encoder.
  // The payload_data must be exactly one encoded generic frame.
  // Packets returned by `NextPacket` will contain raw payload without the
  // generic payload header.
  RtpPacketizerGeneric(rtc::ArrayView<const uint8_t> payload,
                       PayloadSizeLimits limits);

  ~RtpPacketizerGeneric() override;

  RtpPacketizerGeneric(const RtpPacketizerGeneric&) = delete;
  RtpPacketizerGeneric& operator=(const RtpPacketizerGeneric&) = delete;

  size_t NumPackets() const override;

  // Get the next payload.
  // Write payload and set marker bit of the `packet`.
  // Returns true on success, false otherwise.
  bool NextPacket(RtpPacketToSend* packet) override;

 private:
  // Fills header_ and header_size_ members.
  void BuildHeader(const RTPVideoHeader& rtp_video_header);

  uint8_t header_[3];
  size_t header_size_;
  rtc::ArrayView<const uint8_t> remaining_payload_;
  std::vector<int> payload_sizes_;
  std::vector<int>::const_iterator current_packet_;
};
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTP_FORMAT_VIDEO_GENERIC_H_
