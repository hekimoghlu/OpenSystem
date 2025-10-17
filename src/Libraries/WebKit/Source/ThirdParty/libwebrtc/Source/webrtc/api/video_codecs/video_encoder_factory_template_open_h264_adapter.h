/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#ifndef API_VIDEO_CODECS_VIDEO_ENCODER_FACTORY_TEMPLATE_OPEN_H264_ADAPTER_H_
#define API_VIDEO_CODECS_VIDEO_ENCODER_FACTORY_TEMPLATE_OPEN_H264_ADAPTER_H_

#include <memory>
#include <vector>

#include "api/environment/environment.h"
#include "api/video_codecs/scalability_mode.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_encoder.h"
#include "modules/video_coding/codecs/h264/include/h264.h"

namespace webrtc {
// TODO(bugs.webrtc.org/13573): When OpenH264 is no longer a conditional build
//                              target remove #ifdefs.
struct OpenH264EncoderTemplateAdapter {
  static std::vector<SdpVideoFormat> SupportedFormats() {
#if defined(WEBRTC_USE_H264)
    return SupportedH264Codecs(/*add_scalability_modes=*/true);
#else
    return {};
#endif
  }

  static std::unique_ptr<VideoEncoder> CreateEncoder(
      [[maybe_unused]] const Environment& env,
      [[maybe_unused]] const SdpVideoFormat& format) {
#if defined(WEBRTC_USE_H264)
    return CreateH264Encoder(env, H264EncoderSettings::Parse(format));
#else
    return nullptr;
#endif
  }

  static bool IsScalabilityModeSupported(
      [[maybe_unused]] ScalabilityMode scalability_mode) {
#if defined(WEBRTC_USE_H264)
    return H264Encoder::SupportsScalabilityMode(scalability_mode);
#else
    return false;
#endif
  }
};
}  // namespace webrtc

#endif  // API_VIDEO_CODECS_VIDEO_ENCODER_FACTORY_TEMPLATE_OPEN_H264_ADAPTER_H_
