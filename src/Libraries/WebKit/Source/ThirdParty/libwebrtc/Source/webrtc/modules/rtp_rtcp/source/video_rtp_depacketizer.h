/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 19, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_VIDEO_RTP_DEPACKETIZER_H_
#define MODULES_RTP_RTCP_SOURCE_VIDEO_RTP_DEPACKETIZER_H_

#include <stdint.h>

#include <optional>

#include "api/array_view.h"
#include "api/scoped_refptr.h"
#include "api/video/encoded_image.h"
#include "modules/rtp_rtcp/source/rtp_video_header.h"
#include "rtc_base/copy_on_write_buffer.h"

namespace webrtc {

class VideoRtpDepacketizer {
 public:
  struct ParsedRtpPayload {
    RTPVideoHeader video_header;
    rtc::CopyOnWriteBuffer video_payload;
  };

  virtual ~VideoRtpDepacketizer() = default;
  virtual std::optional<ParsedRtpPayload> Parse(
      rtc::CopyOnWriteBuffer rtp_payload) = 0;
  virtual rtc::scoped_refptr<EncodedImageBuffer> AssembleFrame(
      rtc::ArrayView<const rtc::ArrayView<const uint8_t>> rtp_payloads);
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_VIDEO_RTP_DEPACKETIZER_H_
