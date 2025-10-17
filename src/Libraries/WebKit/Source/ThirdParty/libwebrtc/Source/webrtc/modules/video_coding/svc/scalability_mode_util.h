/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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
#ifndef MODULES_VIDEO_CODING_SVC_SCALABILITY_MODE_UTIL_H_
#define MODULES_VIDEO_CODING_SVC_SCALABILITY_MODE_UTIL_H_

#include <optional>

#include "absl/strings/string_view.h"
#include "api/video_codecs/scalability_mode.h"
#include "api/video_codecs/video_codec.h"

namespace webrtc {

enum class ScalabilityModeResolutionRatio {
  kTwoToOne,    // The resolution ratio between spatial layers is 2:1.
  kThreeToTwo,  // The resolution ratio between spatial layers is 1.5:1.
};

static constexpr char kDefaultScalabilityModeStr[] = "L1T2";

RTC_EXPORT std::optional<ScalabilityMode> MakeScalabilityMode(
    int num_spatial_layers,
    int num_temporal_layers,
    InterLayerPredMode inter_layer_pred,
    std::optional<ScalabilityModeResolutionRatio> ratio,
    bool shift);

RTC_EXPORT std::optional<ScalabilityMode> ScalabilityModeFromString(
    absl::string_view scalability_mode_string);

InterLayerPredMode ScalabilityModeToInterLayerPredMode(
    ScalabilityMode scalability_mode);

int ScalabilityModeToNumSpatialLayers(ScalabilityMode scalability_mode);

int ScalabilityModeToNumTemporalLayers(ScalabilityMode scalability_mode);

std::optional<ScalabilityModeResolutionRatio> ScalabilityModeToResolutionRatio(
    ScalabilityMode scalability_mode);

bool ScalabilityModeIsShiftMode(ScalabilityMode scalability_mode);

ScalabilityMode LimitNumSpatialLayers(ScalabilityMode scalability_mode,
                                      int max_spatial_layers);

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_SVC_SCALABILITY_MODE_UTIL_H_
