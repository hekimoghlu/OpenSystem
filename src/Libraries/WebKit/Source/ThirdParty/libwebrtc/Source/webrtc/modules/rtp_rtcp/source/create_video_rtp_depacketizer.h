/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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
#ifndef MODULES_RTP_RTCP_SOURCE_CREATE_VIDEO_RTP_DEPACKETIZER_H_
#define MODULES_RTP_RTCP_SOURCE_CREATE_VIDEO_RTP_DEPACKETIZER_H_

#include <memory>

#include "api/video/video_codec_type.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer.h"

namespace webrtc {

std::unique_ptr<VideoRtpDepacketizer> CreateVideoRtpDepacketizer(
    VideoCodecType codec);

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_CREATE_VIDEO_RTP_DEPACKETIZER_H_
