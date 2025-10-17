/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#ifndef MODULES_VIDEO_CODING_CODECS_VP9_SVC_CONFIG_H_
#define MODULES_VIDEO_CODING_CODECS_VP9_SVC_CONFIG_H_

#include <stddef.h>

#include <vector>

#include "api/video_codecs/spatial_layer.h"
#include "api/video_codecs/video_codec.h"
#include "modules/video_coding/svc/scalable_video_controller.h"

namespace webrtc {

// Uses scalability mode to configure spatial layers.
std::vector<SpatialLayer> GetVp9SvcConfig(VideoCodec& video_codec);

std::vector<SpatialLayer> GetSvcConfig(
    size_t input_width,
    size_t input_height,
    float max_framerate_fps,
    size_t first_active_layer,
    size_t num_spatial_layers,
    size_t num_temporal_layers,
    bool is_screen_sharing,
    std::optional<ScalableVideoController::StreamLayersConfig> config =
        std::nullopt);

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_CODECS_VP9_SVC_CONFIG_H_
