/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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
#ifndef MODULES_RTP_RTCP_SOURCE_VIDEO_RTP_DEPACKETIZER_VP8_H_
#define MODULES_RTP_RTCP_SOURCE_VIDEO_RTP_DEPACKETIZER_VP8_H_

#include <cstdint>
#include <optional>

#include "api/array_view.h"
#include "modules/rtp_rtcp/source/rtp_video_header.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer.h"
#include "rtc_base/copy_on_write_buffer.h"

namespace webrtc {

class VideoRtpDepacketizerVp8 : public VideoRtpDepacketizer {
 public:
  VideoRtpDepacketizerVp8() = default;
  VideoRtpDepacketizerVp8(const VideoRtpDepacketizerVp8&) = delete;
  VideoRtpDepacketizerVp8& operator=(const VideoRtpDepacketizerVp8&) = delete;
  ~VideoRtpDepacketizerVp8() override = default;

  // Parses vp8 rtp payload descriptor.
  // Returns zero on error or vp8 payload header offset on success.
  static int ParseRtpPayload(rtc::ArrayView<const uint8_t> rtp_payload,
                             RTPVideoHeader* video_header);

  std::optional<ParsedRtpPayload> Parse(
      rtc::CopyOnWriteBuffer rtp_payload) override;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_VIDEO_RTP_DEPACKETIZER_VP8_H_
