/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#include "media/engine/internal_encoder_factory.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "api/environment/environment.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "api/video_codecs/video_encoder_factory_template.h"
#if defined(RTC_USE_LIBAOM_AV1_ENCODER)
#include "api/video_codecs/video_encoder_factory_template_libaom_av1_adapter.h"  // nogncheck
#endif
#include "api/video_codecs/video_encoder_factory_template_libvpx_vp8_adapter.h"
#include "api/video_codecs/video_encoder_factory_template_libvpx_vp9_adapter.h"
#if defined(WEBRTC_USE_H264)
#include "api/video_codecs/video_encoder_factory_template_open_h264_adapter.h"  // nogncheck
#endif

namespace webrtc {
namespace {

using Factory =
    VideoEncoderFactoryTemplate<webrtc::LibvpxVp8EncoderTemplateAdapter,
#if defined(WEBRTC_USE_H264)
                                webrtc::OpenH264EncoderTemplateAdapter,
#endif
#if defined(RTC_USE_LIBAOM_AV1_ENCODER)
                                webrtc::LibaomAv1EncoderTemplateAdapter,
#endif
                                webrtc::LibvpxVp9EncoderTemplateAdapter>;
}  // namespace

std::vector<SdpVideoFormat> InternalEncoderFactory::GetSupportedFormats()
    const {
  return Factory().GetSupportedFormats();
}

std::unique_ptr<VideoEncoder> InternalEncoderFactory::Create(
    const Environment& env,
    const SdpVideoFormat& format) {
  auto original_format =
      FuzzyMatchSdpVideoFormat(Factory().GetSupportedFormats(), format);
  return original_format ? Factory().Create(env, *original_format) : nullptr;
}

VideoEncoderFactory::CodecSupport InternalEncoderFactory::QueryCodecSupport(
    const SdpVideoFormat& format,
    std::optional<std::string> scalability_mode) const {
  auto original_format =
      FuzzyMatchSdpVideoFormat(Factory().GetSupportedFormats(), format);
  return original_format
             ? Factory().QueryCodecSupport(*original_format, scalability_mode)
             : VideoEncoderFactory::CodecSupport{.is_supported = false};
}

}  // namespace webrtc
