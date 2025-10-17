/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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
#include "modules/video_coding/deprecated/packet.h"

#include "api/rtp_headers.h"

namespace webrtc {

VCMPacket::VCMPacket()
    : payloadType(0),
      timestamp(0),
      ntp_time_ms_(0),
      seqNum(0),
      dataPtr(NULL),
      sizeBytes(0),
      markerBit(false),
      timesNacked(-1),
      completeNALU(kNaluUnset),
      insertStartCode(false),
      video_header() {
}

VCMPacket::VCMPacket(const uint8_t* ptr,
                     size_t size,
                     const RTPHeader& rtp_header,
                     const RTPVideoHeader& videoHeader,
                     int64_t ntp_time_ms,
                     Timestamp receive_time)
    : payloadType(rtp_header.payloadType),
      timestamp(rtp_header.timestamp),
      ntp_time_ms_(ntp_time_ms),
      seqNum(rtp_header.sequenceNumber),
      dataPtr(ptr),
      sizeBytes(size),
      markerBit(rtp_header.markerBit),
      timesNacked(-1),
      completeNALU(kNaluIncomplete),
      insertStartCode(videoHeader.codec == kVideoCodecH264 &&
                      videoHeader.is_first_packet_in_frame),
      video_header(videoHeader),
      packet_info(rtp_header, receive_time) {
  if (is_first_packet_in_frame() && markerBit) {
    completeNALU = kNaluComplete;
  } else if (is_first_packet_in_frame()) {
    completeNALU = kNaluStart;
  } else if (markerBit) {
    completeNALU = kNaluEnd;
  } else {
    completeNALU = kNaluIncomplete;
  }

  // Playout decisions are made entirely based on first packet in a frame.
  if (!is_first_packet_in_frame()) {
    video_header.playout_delay = std::nullopt;
  }
}

VCMPacket::~VCMPacket() = default;

}  // namespace webrtc
