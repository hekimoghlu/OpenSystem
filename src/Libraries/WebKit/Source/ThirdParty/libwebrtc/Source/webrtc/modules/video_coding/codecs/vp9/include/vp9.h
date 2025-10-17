/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#ifndef MODULES_VIDEO_CODING_CODECS_VP9_INCLUDE_VP9_H_
#define MODULES_VIDEO_CODING_CODECS_VP9_INCLUDE_VP9_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "api/environment/environment.h"
#include "api/video_codecs/scalability_mode.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_encoder.h"
#include "api/video_codecs/vp9_profile.h"
#include "modules/video_coding/include/video_codec_interface.h"

namespace webrtc {

// Returns a vector with all supported internal VP9 profiles that we can
// negotiate in SDP, in order of preference.
std::vector<SdpVideoFormat> SupportedVP9Codecs(
    bool add_scalability_modes = false);

// Returns a vector with all supported internal VP9 decode profiles in order of
// preference. These will be availble for receive-only connections.
std::vector<SdpVideoFormat> SupportedVP9DecoderCodecs();

struct Vp9EncoderSettings {
  VP9Profile profile = VP9Profile::kProfile0;
};
absl::Nonnull<std::unique_ptr<VideoEncoder>> CreateVp9Encoder(
    const Environment& env,
    Vp9EncoderSettings settings = {});

class VP9Encoder {
 public:
  static bool SupportsScalabilityMode(ScalabilityMode scalability_mode);
};

class VP9Decoder : public VideoDecoder {
 public:
  static std::unique_ptr<VP9Decoder> Create();

  ~VP9Decoder() override {}
};
}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_CODECS_VP9_INCLUDE_VP9_H_
