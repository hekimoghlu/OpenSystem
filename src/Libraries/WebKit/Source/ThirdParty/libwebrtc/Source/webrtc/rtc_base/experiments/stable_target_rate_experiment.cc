/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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
#include "rtc_base/experiments/stable_target_rate_experiment.h"

#include "api/field_trials_view.h"
#include "rtc_base/experiments/field_trial_parser.h"

namespace webrtc {
namespace {
constexpr char kFieldTrialName[] = "WebRTC-StableTargetRate";
}  // namespace

StableTargetRateExperiment::StableTargetRateExperiment(
    const FieldTrialsView& key_value_config)
    : enabled_("enabled", false),
      video_hysteresis_factor_("video_hysteresis_factor",
                               /*default_value=*/1.2),
      screenshare_hysteresis_factor_("screenshare_hysteresis_factor",
                                     /*default_value=*/1.35) {
  ParseFieldTrial(
      {&enabled_, &video_hysteresis_factor_, &screenshare_hysteresis_factor_},
      key_value_config.Lookup(kFieldTrialName));
}

StableTargetRateExperiment::StableTargetRateExperiment(
    const StableTargetRateExperiment&) = default;
StableTargetRateExperiment::StableTargetRateExperiment(
    StableTargetRateExperiment&&) = default;

bool StableTargetRateExperiment::IsEnabled() const {
  return enabled_.Get();
}

double StableTargetRateExperiment::GetVideoHysteresisFactor() const {
  return video_hysteresis_factor_.Get();
}

double StableTargetRateExperiment::GetScreenshareHysteresisFactor() const {
  return screenshare_hysteresis_factor_.Get();
}

}  // namespace webrtc
