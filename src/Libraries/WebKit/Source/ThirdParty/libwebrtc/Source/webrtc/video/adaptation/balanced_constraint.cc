/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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
#include "video/adaptation/balanced_constraint.h"

#include <string>
#include <utility>

#include "api/sequence_checker.h"

namespace webrtc {

BalancedConstraint::BalancedConstraint(
    DegradationPreferenceProvider* degradation_preference_provider,
    const FieldTrialsView& field_trials)
    : encoder_target_bitrate_bps_(std::nullopt),
      balanced_settings_(field_trials),
      degradation_preference_provider_(degradation_preference_provider) {
  RTC_DCHECK(degradation_preference_provider_);
  sequence_checker_.Detach();
}

void BalancedConstraint::OnEncoderTargetBitrateUpdated(
    std::optional<uint32_t> encoder_target_bitrate_bps) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  encoder_target_bitrate_bps_ = std::move(encoder_target_bitrate_bps);
}

bool BalancedConstraint::IsAdaptationUpAllowed(
    const VideoStreamInputState& input_state,
    const VideoSourceRestrictions& restrictions_before,
    const VideoSourceRestrictions& restrictions_after) const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  // Don't adapt if BalancedDegradationSettings applies and determines this will
  // exceed bitrate constraints.
  if (degradation_preference_provider_->degradation_preference() ==
      DegradationPreference::BALANCED) {
    int frame_size_pixels = input_state.single_active_stream_pixels().value_or(
        input_state.frame_size_pixels().value());
    if (!balanced_settings_.CanAdaptUp(
            input_state.video_codec_type(), frame_size_pixels,
            encoder_target_bitrate_bps_.value_or(0))) {
      return false;
    }
    if (DidIncreaseResolution(restrictions_before, restrictions_after) &&
        !balanced_settings_.CanAdaptUpResolution(
            input_state.video_codec_type(), frame_size_pixels,
            encoder_target_bitrate_bps_.value_or(0))) {
      return false;
    }
  }
  return true;
}

}  // namespace webrtc
