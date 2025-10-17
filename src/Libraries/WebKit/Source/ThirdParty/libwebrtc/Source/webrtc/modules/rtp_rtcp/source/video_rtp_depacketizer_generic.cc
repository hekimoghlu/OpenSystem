/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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
#include "modules/rtp_rtcp/source/video_rtp_depacketizer_generic.h"

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <utility>

#include "modules/rtp_rtcp/source/rtp_video_header.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer.h"
#include "rtc_base/copy_on_write_buffer.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace {
constexpr uint8_t kKeyFrameBit = 0b0000'0001;
constexpr uint8_t kFirstPacketBit = 0b0000'0010;
// If this bit is set, there will be an extended header contained in this
// packet. This was added later so old clients will not send this.
constexpr uint8_t kExtendedHeaderBit = 0b0000'0100;

constexpr size_t kGenericHeaderLength = 1;
constexpr size_t kExtendedHeaderLength = 2;
}  // namespace

std::optional<VideoRtpDepacketizer::ParsedRtpPayload>
VideoRtpDepacketizerGeneric::Parse(rtc::CopyOnWriteBuffer rtp_payload) {
  if (rtp_payload.size() == 0) {
    RTC_LOG(LS_WARNING) << "Empty payload.";
    return std::nullopt;
  }
  std::optional<ParsedRtpPayload> parsed(std::in_place);
  const uint8_t* payload_data = rtp_payload.cdata();

  uint8_t generic_header = payload_data[0];
  size_t offset = kGenericHeaderLength;

  parsed->video_header.frame_type = (generic_header & kKeyFrameBit)
                                        ? VideoFrameType::kVideoFrameKey
                                        : VideoFrameType::kVideoFrameDelta;
  parsed->video_header.is_first_packet_in_frame =
      (generic_header & kFirstPacketBit) != 0;
  parsed->video_header.codec = kVideoCodecGeneric;
  parsed->video_header.width = 0;
  parsed->video_header.height = 0;

  if (generic_header & kExtendedHeaderBit) {
    if (rtp_payload.size() < offset + kExtendedHeaderLength) {
      RTC_LOG(LS_WARNING) << "Too short payload for generic header.";
      return std::nullopt;
    }
    parsed->video_header.video_type_header
        .emplace<RTPVideoHeaderLegacyGeneric>()
        .picture_id = ((payload_data[1] & 0x7F) << 8) | payload_data[2];
    offset += kExtendedHeaderLength;
  }

  parsed->video_payload =
      rtp_payload.Slice(offset, rtp_payload.size() - offset);
  return parsed;
}
}  // namespace webrtc
