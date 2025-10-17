/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#ifndef API_VIDEO_CODECS_VIDEO_ENCODER_FACTORY_TEMPLATE_LIBAOM_AV1_ADAPTER_H_
#define API_VIDEO_CODECS_VIDEO_ENCODER_FACTORY_TEMPLATE_LIBAOM_AV1_ADAPTER_H_

#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "api/environment/environment.h"
#include "api/video_codecs/scalability_mode.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_encoder.h"
#include "modules/video_coding/codecs/av1/av1_svc_config.h"
#include "modules/video_coding/codecs/av1/libaom_av1_encoder.h"

namespace webrtc {
struct LibaomAv1EncoderTemplateAdapter {
  static std::vector<SdpVideoFormat> SupportedFormats() {
    absl::InlinedVector<ScalabilityMode, kScalabilityModeCount>
        scalability_modes = LibaomAv1EncoderSupportedScalabilityModes();
    return {SdpVideoFormat(SdpVideoFormat::AV1Profile0(), scalability_modes)};
  }

  static std::unique_ptr<VideoEncoder> CreateEncoder(
      const Environment& env,
      const SdpVideoFormat& /* format */) {
    return CreateLibaomAv1Encoder(env);
  }

  static bool IsScalabilityModeSupported(ScalabilityMode scalability_mode) {
    return LibaomAv1EncoderSupportsScalabilityMode(scalability_mode);
  }
};

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_VIDEO_ENCODER_FACTORY_TEMPLATE_LIBAOM_AV1_ADAPTER_H_
