/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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
#include "video/adaptation/bitrate_constraint.h"

#include <utility>
#include <vector>

#include "api/sequence_checker.h"
#include "call/adaptation/video_stream_adapter.h"
#include "video/adaptation/video_stream_encoder_resource_manager.h"

namespace webrtc {

BitrateConstraint::BitrateConstraint()
    : encoder_settings_(std::nullopt),
      encoder_target_bitrate_bps_(std::nullopt) {
  sequence_checker_.Detach();
}

void BitrateConstraint::OnEncoderSettingsUpdated(
    std::optional<EncoderSettings> encoder_settings) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  encoder_settings_ = std::move(encoder_settings);
}

void BitrateConstraint::OnEncoderTargetBitrateUpdated(
    std::optional<uint32_t> encoder_target_bitrate_bps) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  encoder_target_bitrate_bps_ = std::move(encoder_target_bitrate_bps);
}

// Checks if resolution is allowed to adapt up based on the current bitrate and
// ResolutionBitrateLimits.min_start_bitrate_bps for the next higher resolution.
// Bitrate limits usage is restricted to a single active stream/layer (e.g. when
// quality scaling is enabled).
bool BitrateConstraint::IsAdaptationUpAllowed(
    const VideoStreamInputState& input_state,
    const VideoSourceRestrictions& restrictions_before,
    const VideoSourceRestrictions& restrictions_after) const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  // Make sure bitrate limits are not violated.
  if (DidIncreaseResolution(restrictions_before, restrictions_after)) {
    if (!encoder_settings_.has_value()) {
      return true;
    }

    uint32_t bitrate_bps = encoder_target_bitrate_bps_.value_or(0);
    if (bitrate_bps == 0) {
      return true;
    }

    if (VideoStreamEncoderResourceManager::IsSimulcastOrMultipleSpatialLayers(
            encoder_settings_->encoder_config(),
            encoder_settings_->video_codec())) {
      // Resolution bitrate limits usage is restricted to singlecast.
      return true;
    }

    std::optional<int> current_frame_size_px =
        input_state.single_active_stream_pixels();
    if (!current_frame_size_px.has_value()) {
      return true;
    }

    std::optional<VideoEncoder::ResolutionBitrateLimits> bitrate_limits =
        encoder_settings_->encoder_info().GetEncoderBitrateLimitsForResolution(
            // Need some sort of expected resulting pixels to be used
            // instead of unrestricted.
            GetHigherResolutionThan(*current_frame_size_px));

    if (bitrate_limits.has_value()) {
      RTC_DCHECK_GE(bitrate_limits->frame_size_pixels, *current_frame_size_px);
      return bitrate_bps >=
             static_cast<uint32_t>(bitrate_limits->min_start_bitrate_bps);
    }
  }
  return true;
}

}  // namespace webrtc
