/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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
/*
 * This file contains the declaration of the VP8 packetizer class.
 * A packetizer object is created for each encoded video frame. The
 * constructor is called with the payload data and size,
 * together with the fragmentation information and a packetizer mode
 * of choice. Alternatively, if no fragmentation info is available, the
 * second constructor can be used with only payload data and size; in that
 * case the mode kEqualSize is used.
 *
 * After creating the packetizer, the method NextPacket is called
 * repeatedly to get all packets for the frame. The method returns
 * false as long as there are more packets left to fetch.
 */

#ifndef MODULES_RTP_RTCP_SOURCE_RTP_FORMAT_VP8_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_FORMAT_VP8_H_

#include <stddef.h>

#include <cstdint>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "api/array_view.h"
#include "modules/rtp_rtcp/source/rtp_format.h"
#include "modules/rtp_rtcp/source/rtp_packet_to_send.h"
#include "modules/video_coding/codecs/vp8/include/vp8_globals.h"

namespace webrtc {

// Packetizer for VP8.
class RtpPacketizerVp8 : public RtpPacketizer {
 public:
  // Initialize with payload from encoder.
  // The payload_data must be exactly one encoded VP8 frame.
  RtpPacketizerVp8(rtc::ArrayView<const uint8_t> payload,
                   PayloadSizeLimits limits,
                   const RTPVideoHeaderVP8& hdr_info);

  ~RtpPacketizerVp8() override;

  RtpPacketizerVp8(const RtpPacketizerVp8&) = delete;
  RtpPacketizerVp8& operator=(const RtpPacketizerVp8&) = delete;

  size_t NumPackets() const override;

  // Get the next payload with VP8 payload header.
  // Write payload and set marker bit of the `packet`.
  // Returns true on success, false otherwise.
  bool NextPacket(RtpPacketToSend* packet) override;

 private:
  // VP8 header can use up to 6 bytes.
  using RawHeader = absl::InlinedVector<uint8_t, 6>;
  static RawHeader BuildHeader(const RTPVideoHeaderVP8& header);

  RawHeader hdr_;
  rtc::ArrayView<const uint8_t> remaining_payload_;
  std::vector<int> payload_sizes_;
  std::vector<int>::const_iterator current_packet_;
};

}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTP_FORMAT_VP8_H_
