/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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
#ifndef MODULES_VIDEO_CODING_CODECS_AV1_LIBAOM_AV1_ENCODER_H_
#define MODULES_VIDEO_CODING_CODECS_AV1_LIBAOM_AV1_ENCODER_H_

#include <map>
#include <memory>

#include "absl/base/nullability.h"
#include "api/environment/environment.h"
#include "api/video_codecs/video_encoder.h"

namespace webrtc {

struct LibaomAv1EncoderSettings {
  // A map of max pixel count --> cpu speed.
  std::map<int, int> max_pixel_count_to_cpu_speed;
};
absl::Nonnull<std::unique_ptr<VideoEncoder>> CreateLibaomAv1Encoder(
    const Environment& env,
    LibaomAv1EncoderSettings settings = {});

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_CODECS_AV1_LIBAOM_AV1_ENCODER_H_
