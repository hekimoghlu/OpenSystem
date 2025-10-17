/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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
#include "modules/rtp_rtcp/source/rtp_packet_to_send.h"

#include <cstdint>

#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"

namespace webrtc {

RtpPacketToSend::RtpPacketToSend(const ExtensionManager* extensions)
    : RtpPacket(extensions) {}
RtpPacketToSend::RtpPacketToSend(const ExtensionManager* extensions,
                                 size_t capacity)
    : RtpPacket(extensions, capacity) {}
RtpPacketToSend::RtpPacketToSend(const RtpPacketToSend& packet) = default;
RtpPacketToSend::RtpPacketToSend(RtpPacketToSend&& packet) = default;

RtpPacketToSend& RtpPacketToSend::operator=(const RtpPacketToSend& packet) =
    default;
RtpPacketToSend& RtpPacketToSend::operator=(RtpPacketToSend&& packet) = default;

RtpPacketToSend::~RtpPacketToSend() = default;

void RtpPacketToSend::set_packet_type(RtpPacketMediaType type) {
  if (packet_type_ == RtpPacketMediaType::kAudio) {
    original_packet_type_ = OriginalType::kAudio;
  } else if (packet_type_ == RtpPacketMediaType::kVideo) {
    original_packet_type_ = OriginalType::kVideo;
  }
  packet_type_ = type;
}

}  // namespace webrtc
