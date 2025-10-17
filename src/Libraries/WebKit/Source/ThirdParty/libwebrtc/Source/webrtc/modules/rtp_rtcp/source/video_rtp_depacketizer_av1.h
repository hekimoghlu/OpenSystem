/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
#ifndef MODULES_RTP_RTCP_SOURCE_VIDEO_RTP_DEPACKETIZER_AV1_H_
#define MODULES_RTP_RTCP_SOURCE_VIDEO_RTP_DEPACKETIZER_AV1_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "api/array_view.h"
#include "api/scoped_refptr.h"
#include "api/video/encoded_image.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer.h"
#include "rtc_base/copy_on_write_buffer.h"

namespace webrtc {

class VideoRtpDepacketizerAv1 : public VideoRtpDepacketizer {
 public:
  VideoRtpDepacketizerAv1() = default;
  VideoRtpDepacketizerAv1(const VideoRtpDepacketizerAv1&) = delete;
  VideoRtpDepacketizerAv1& operator=(const VideoRtpDepacketizerAv1&) = delete;
  ~VideoRtpDepacketizerAv1() override = default;

  rtc::scoped_refptr<EncodedImageBuffer> AssembleFrame(
      rtc::ArrayView<const rtc::ArrayView<const uint8_t>> rtp_payloads)
      override;

  std::optional<ParsedRtpPayload> Parse(
      rtc::CopyOnWriteBuffer rtp_payload) override;
};

}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_VIDEO_RTP_DEPACKETIZER_AV1_H_
