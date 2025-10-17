/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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
#ifndef VIDEO_ADAPTATION_BALANCED_CONSTRAINT_H_
#define VIDEO_ADAPTATION_BALANCED_CONSTRAINT_H_

#include <optional>
#include <string>

#include "api/field_trials_view.h"
#include "api/sequence_checker.h"
#include "call/adaptation/adaptation_constraint.h"
#include "call/adaptation/degradation_preference_provider.h"
#include "rtc_base/experiments/balanced_degradation_settings.h"
#include "rtc_base/system/no_unique_address.h"

namespace webrtc {

class BalancedConstraint : public AdaptationConstraint {
 public:
  BalancedConstraint(
      DegradationPreferenceProvider* degradation_preference_provider,
      const FieldTrialsView& field_trials);
  ~BalancedConstraint() override = default;

  void OnEncoderTargetBitrateUpdated(
      std::optional<uint32_t> encoder_target_bitrate_bps);

  // AdaptationConstraint implementation.
  std::string Name() const override { return "BalancedConstraint"; }
  bool IsAdaptationUpAllowed(
      const VideoStreamInputState& input_state,
      const VideoSourceRestrictions& restrictions_before,
      const VideoSourceRestrictions& restrictions_after) const override;

 private:
  RTC_NO_UNIQUE_ADDRESS SequenceChecker sequence_checker_;
  std::optional<uint32_t> encoder_target_bitrate_bps_
      RTC_GUARDED_BY(&sequence_checker_);
  const BalancedDegradationSettings balanced_settings_;
  const DegradationPreferenceProvider* degradation_preference_provider_;
};

}  // namespace webrtc

#endif  // VIDEO_ADAPTATION_BALANCED_CONSTRAINT_H_
