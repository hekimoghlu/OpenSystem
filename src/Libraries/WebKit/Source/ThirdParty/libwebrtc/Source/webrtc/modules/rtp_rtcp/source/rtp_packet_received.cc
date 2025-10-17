/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
#include "modules/rtp_rtcp/source/rtp_packet_received.h"

#include <stddef.h>

#include <cstdint>
#include <vector>

#include "modules/rtp_rtcp/source/rtp_header_extensions.h"
#include "rtc_base/numerics/safe_conversions.h"

namespace webrtc {

RtpPacketReceived::RtpPacketReceived() = default;
RtpPacketReceived::RtpPacketReceived(
    const ExtensionManager* extensions,
    webrtc::Timestamp arrival_time /*= webrtc::Timestamp::MinusInfinity()*/)
    : RtpPacket(extensions), arrival_time_(arrival_time) {}
RtpPacketReceived::RtpPacketReceived(const RtpPacketReceived& packet) = default;
RtpPacketReceived::RtpPacketReceived(RtpPacketReceived&& packet) = default;

RtpPacketReceived& RtpPacketReceived::operator=(
    const RtpPacketReceived& packet) = default;
RtpPacketReceived& RtpPacketReceived::operator=(RtpPacketReceived&& packet) =
    default;

RtpPacketReceived::~RtpPacketReceived() {}

void RtpPacketReceived::GetHeader(RTPHeader* header) const {
  header->markerBit = Marker();
  header->payloadType = PayloadType();
  header->sequenceNumber = SequenceNumber();
  header->timestamp = Timestamp();
  header->ssrc = Ssrc();
  std::vector<uint32_t> csrcs = Csrcs();
  header->numCSRCs = rtc::dchecked_cast<uint8_t>(csrcs.size());
  for (size_t i = 0; i < csrcs.size(); ++i) {
    header->arrOfCSRCs[i] = csrcs[i];
  }
  header->paddingLength = padding_size();
  header->headerLength = headers_size();
  header->extension.hasTransmissionTimeOffset =
      GetExtension<TransmissionOffset>(
          &header->extension.transmissionTimeOffset);
  header->extension.hasAbsoluteSendTime =
      GetExtension<AbsoluteSendTime>(&header->extension.absoluteSendTime);
  header->extension.absolute_capture_time =
      GetExtension<AbsoluteCaptureTimeExtension>();
  header->extension.hasTransportSequenceNumber =
      GetExtension<TransportSequenceNumberV2>(
          &header->extension.transportSequenceNumber,
          &header->extension.feedback_request) ||
      GetExtension<TransportSequenceNumber>(
          &header->extension.transportSequenceNumber);
  header->extension.set_audio_level(GetExtension<AudioLevelExtension>());
  header->extension.hasVideoRotation =
      GetExtension<VideoOrientation>(&header->extension.videoRotation);
  header->extension.hasVideoContentType =
      GetExtension<VideoContentTypeExtension>(
          &header->extension.videoContentType);
  header->extension.has_video_timing =
      GetExtension<VideoTimingExtension>(&header->extension.video_timing);
  GetExtension<RtpStreamId>(&header->extension.stream_id);
  GetExtension<RepairedRtpStreamId>(&header->extension.repaired_stream_id);
  GetExtension<RtpMid>(&header->extension.mid);
  GetExtension<PlayoutDelayLimits>(&header->extension.playout_delay);
  header->extension.color_space = GetExtension<ColorSpaceExtension>();
}

}  // namespace webrtc
