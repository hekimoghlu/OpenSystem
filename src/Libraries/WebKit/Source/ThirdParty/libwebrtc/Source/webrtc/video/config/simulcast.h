/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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
#ifndef VIDEO_CONFIG_SIMULCAST_H_
#define VIDEO_CONFIG_SIMULCAST_H_

#include <stddef.h>

#include <vector>

#include "api/array_view.h"
#include "api/field_trials_view.h"
#include "api/units/data_rate.h"
#include "api/video/resolution.h"
#include "video/config/video_encoder_config.h"

namespace cricket {

// Gets the total maximum bitrate for the `streams`.
webrtc::DataRate GetTotalMaxBitrate(
    const std::vector<webrtc::VideoStream>& streams);

// Adds any bitrate of `max_bitrate` that is above the total maximum bitrate for
// the `layers` to the highest quality layer.
void BoostMaxSimulcastLayer(webrtc::DataRate max_bitrate,
                            std::vector<webrtc::VideoStream>* layers);

// Returns number of simulcast streams. The value depends on the resolution and
// is restricted to the range from `min_num_layers` to `max_num_layers`,
// inclusive.
size_t LimitSimulcastLayerCount(size_t min_num_layers,
                                size_t max_num_layers,
                                int width,
                                int height,
                                const webrtc::FieldTrialsView& trials,
                                webrtc::VideoCodecType codec);

// Gets simulcast settings.
std::vector<webrtc::VideoStream> GetSimulcastConfig(
    rtc::ArrayView<const webrtc::Resolution> resolutions,
    bool is_screenshare_with_conference_mode,
    bool temporal_layers_supported,
    const webrtc::FieldTrialsView& trials,
    webrtc::VideoCodecType codec);

}  // namespace cricket

#endif  // VIDEO_CONFIG_SIMULCAST_H_
