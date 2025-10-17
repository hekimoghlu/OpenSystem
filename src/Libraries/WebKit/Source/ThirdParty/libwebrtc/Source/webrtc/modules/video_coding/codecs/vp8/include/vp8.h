/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 23, 2025.
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
#ifndef MODULES_VIDEO_CODING_CODECS_VP8_INCLUDE_VP8_H_
#define MODULES_VIDEO_CODING_CODECS_VP8_INCLUDE_VP8_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "api/environment/environment.h"
#include "api/video_codecs/video_encoder.h"
#include "modules/video_coding/include/video_codec_interface.h"

namespace webrtc {

struct Vp8EncoderSettings {
  // Allows for overriding the resolution/bitrate limits exposed through
  // VideoEncoder::GetEncoderInfo(). No override is done if empty.
  std::vector<VideoEncoder::ResolutionBitrateLimits> resolution_bitrate_limits;
};
absl::Nonnull<std::unique_ptr<VideoEncoder>> CreateVp8Encoder(
    const Environment& env,
    Vp8EncoderSettings settings = {});

std::unique_ptr<VideoDecoder> CreateVp8Decoder(const Environment& env);

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_CODECS_VP8_INCLUDE_VP8_H_
