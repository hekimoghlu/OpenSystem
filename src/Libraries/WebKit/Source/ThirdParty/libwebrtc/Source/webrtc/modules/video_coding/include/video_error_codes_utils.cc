/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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
#include "modules/video_coding/include/video_error_codes_utils.h"

#include "modules/video_coding/include/video_error_codes.h"

namespace webrtc {

const char* WebRtcVideoCodecErrorToString(int32_t error_code) {
  switch (error_code) {
    case WEBRTC_VIDEO_CODEC_TARGET_BITRATE_OVERSHOOT:
      return "WEBRTC_VIDEO_CODEC_TARGET_BITRATE_OVERSHOOT";
    case WEBRTC_VIDEO_CODEC_OK_REQUEST_KEYFRAME:
      return "WEBRTC_VIDEO_CODEC_OK_REQUEST_KEYFRAME";
    case WEBRTC_VIDEO_CODEC_NO_OUTPUT:
      return "WEBRTC_VIDEO_CODEC_NO_OUTPUT";
    case WEBRTC_VIDEO_CODEC_ERROR:
      return "WEBRTC_VIDEO_CODEC_ERROR";
    case WEBRTC_VIDEO_CODEC_MEMORY:
      return "WEBRTC_VIDEO_CODEC_MEMORY";
    case WEBRTC_VIDEO_CODEC_ERR_PARAMETER:
      return "WEBRTC_VIDEO_CODEC_ERR_PARAMETER";
    case WEBRTC_VIDEO_CODEC_TIMEOUT:
      return "WEBRTC_VIDEO_CODEC_TIMEOUT";
    case WEBRTC_VIDEO_CODEC_UNINITIALIZED:
      return "WEBRTC_VIDEO_CODEC_UNINITIALIZED";
    case WEBRTC_VIDEO_CODEC_FALLBACK_SOFTWARE:
      return "WEBRTC_VIDEO_CODEC_FALLBACK_SOFTWARE";
    case WEBRTC_VIDEO_CODEC_ERR_SIMULCAST_PARAMETERS_NOT_SUPPORTED:
      return "WEBRTC_VIDEO_CODEC_ERR_SIMULCAST_PARAMETERS_NOT_SUPPORTED";
    case WEBRTC_VIDEO_CODEC_ENCODER_FAILURE:
      return "WEBRTC_VIDEO_CODEC_ENCODER_FAILURE";
    default:
      return "WEBRTC_VIDEO_CODEC_UNKNOWN";
  }
}

}  // namespace webrtc
