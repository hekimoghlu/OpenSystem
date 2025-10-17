/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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
#include "modules/rtp_rtcp/source/create_video_rtp_depacketizer.h"

#include <memory>

#include "api/video/video_codec_type.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer_av1.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer_generic.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer_h264.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer_vp8.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer_vp9.h"
#ifdef RTC_ENABLE_H265
#include "modules/rtp_rtcp/source/video_rtp_depacketizer_h265.h"
#endif

namespace webrtc {

std::unique_ptr<VideoRtpDepacketizer> CreateVideoRtpDepacketizer(
    VideoCodecType codec) {
  switch (codec) {
    case kVideoCodecH264:
      return std::make_unique<VideoRtpDepacketizerH264>();
    case kVideoCodecVP8:
      return std::make_unique<VideoRtpDepacketizerVp8>();
    case kVideoCodecVP9:
      return std::make_unique<VideoRtpDepacketizerVp9>();
    case kVideoCodecAV1:
      return std::make_unique<VideoRtpDepacketizerAv1>();
    case kVideoCodecH265:
#ifdef RTC_ENABLE_H265
      return std::make_unique<VideoRtpDepacketizerH265>();
#else
      return nullptr;
#endif
    case kVideoCodecGeneric:
      return std::make_unique<VideoRtpDepacketizerGeneric>();
  }
  RTC_CHECK_NOTREACHED();
}

}  // namespace webrtc
