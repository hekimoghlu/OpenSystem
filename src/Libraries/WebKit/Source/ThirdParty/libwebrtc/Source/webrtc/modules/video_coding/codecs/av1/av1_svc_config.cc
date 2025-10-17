/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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
#include "modules/video_coding/codecs/av1/av1_svc_config.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "modules/video_coding/svc/create_scalability_structure.h"
#include "modules/video_coding/svc/scalability_mode_util.h"
#include "modules/video_coding/svc/scalable_video_controller.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/strings/string_builder.h"

namespace webrtc {
namespace {
const int kMinAv1SpatialLayerLongSideLength = 240;
const int kMinAv1SpatialLayerShortSideLength = 135;

int GetLimitedNumSpatialLayers(int width, int height) {
  const bool is_landscape = width >= height;
  const int min_width = is_landscape ? kMinAv1SpatialLayerLongSideLength
                                     : kMinAv1SpatialLayerShortSideLength;
  const int min_height = is_landscape ? kMinAv1SpatialLayerShortSideLength
                                      : kMinAv1SpatialLayerLongSideLength;
  const int num_layers_fit_horz = static_cast<int>(
      std::floor(1 + std::max(0.0f, std::log2(1.0f * width / min_width))));
  const int num_layers_fit_vert = static_cast<int>(
      std::floor(1 + std::max(0.0f, std::log2(1.0f * height / min_height))));
  return std::min(num_layers_fit_horz, num_layers_fit_vert);
}

std::optional<ScalabilityMode> BuildScalabilityMode(int num_temporal_layers,
                                                    int num_spatial_layers) {
  char name[20];
  rtc::SimpleStringBuilder ss(name);
  ss << "L" << num_spatial_layers << "T" << num_temporal_layers;
  if (num_spatial_layers > 1) {
    ss << "_KEY";
  }

  return ScalabilityModeFromString(name);
}
}  // namespace

absl::InlinedVector<ScalabilityMode, kScalabilityModeCount>
LibaomAv1EncoderSupportedScalabilityModes() {
  absl::InlinedVector<ScalabilityMode, kScalabilityModeCount> scalability_modes;
  for (ScalabilityMode scalability_mode : kAllScalabilityModes) {
    if (ScalabilityStructureConfig(scalability_mode) != std::nullopt) {
      scalability_modes.push_back(scalability_mode);
    }
  }
  return scalability_modes;
}

bool LibaomAv1EncoderSupportsScalabilityMode(ScalabilityMode scalability_mode) {
  // For libaom AV1, the scalability mode is supported if we can create the
  // scalability structure.
  return ScalabilityStructureConfig(scalability_mode) != std::nullopt;
}

bool SetAv1SvcConfig(VideoCodec& video_codec,
                     int num_temporal_layers,
                     int num_spatial_layers) {
  RTC_DCHECK_EQ(video_codec.codecType, kVideoCodecAV1);

  std::optional<ScalabilityMode> scalability_mode =
      video_codec.GetScalabilityMode();
  if (!scalability_mode.has_value()) {
    scalability_mode =
        BuildScalabilityMode(num_temporal_layers, num_spatial_layers);
    if (!scalability_mode) {
      RTC_LOG(LS_WARNING) << "Scalability mode is not set, using 'L1T1'.";
      scalability_mode = ScalabilityMode::kL1T1;
    }
  }

  bool requested_single_spatial_layer =
      ScalabilityModeToNumSpatialLayers(*scalability_mode) == 1;

  if (ScalabilityMode reduced = LimitNumSpatialLayers(
          *scalability_mode,
          GetLimitedNumSpatialLayers(video_codec.width, video_codec.height));
      *scalability_mode != reduced) {
    RTC_LOG(LS_WARNING) << "Reduced number of spatial layers from "
                        << ScalabilityModeToString(*scalability_mode) << " to "
                        << ScalabilityModeToString(reduced);
    scalability_mode = reduced;
  }

  std::unique_ptr<ScalableVideoController> structure =
      CreateScalabilityStructure(*scalability_mode);
  if (structure == nullptr) {
    RTC_LOG(LS_WARNING) << "Failed to create structure "
                        << static_cast<int>(*scalability_mode);
    return false;
  }

  video_codec.SetScalabilityMode(*scalability_mode);

  ScalableVideoController::StreamLayersConfig info = structure->StreamConfig();
  for (int sl_idx = 0; sl_idx < info.num_spatial_layers; ++sl_idx) {
    SpatialLayer& spatial_layer = video_codec.spatialLayers[sl_idx];
    spatial_layer.width = video_codec.width * info.scaling_factor_num[sl_idx] /
                          info.scaling_factor_den[sl_idx];
    spatial_layer.height = video_codec.height *
                           info.scaling_factor_num[sl_idx] /
                           info.scaling_factor_den[sl_idx];
    spatial_layer.maxFramerate = video_codec.maxFramerate;
    spatial_layer.numberOfTemporalLayers = info.num_temporal_layers;
    spatial_layer.active = true;
  }

  if (requested_single_spatial_layer) {
    SpatialLayer& spatial_layer = video_codec.spatialLayers[0];
    spatial_layer.minBitrate = video_codec.minBitrate;
    spatial_layer.maxBitrate = video_codec.maxBitrate;
    spatial_layer.targetBitrate =
        (video_codec.minBitrate + video_codec.maxBitrate) / 2;
    return true;
  }

  for (int sl_idx = 0; sl_idx < info.num_spatial_layers; ++sl_idx) {
    SpatialLayer& spatial_layer = video_codec.spatialLayers[sl_idx];
    const int num_pixels = spatial_layer.width * spatial_layer.height;
    int min_bitrate_kbps = (480.0 * std::sqrt(num_pixels) - 95'000.0) / 1000.0;
    spatial_layer.minBitrate = std::max(min_bitrate_kbps, 20);
    spatial_layer.maxBitrate = 50 + static_cast<int>(1.6 * num_pixels / 1000.0);
    spatial_layer.targetBitrate =
        (spatial_layer.minBitrate + spatial_layer.maxBitrate) / 2;
  }
  return true;
}

}  // namespace webrtc
