/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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
#ifndef RTC_BASE_EXPERIMENTS_STABLE_TARGET_RATE_EXPERIMENT_H_
#define RTC_BASE_EXPERIMENTS_STABLE_TARGET_RATE_EXPERIMENT_H_

#include "api/field_trials_view.h"
#include "rtc_base/experiments/field_trial_parser.h"

namespace webrtc {

class StableTargetRateExperiment {
 public:
  explicit StableTargetRateExperiment(const FieldTrialsView& field_trials);
  StableTargetRateExperiment(const StableTargetRateExperiment&);
  StableTargetRateExperiment(StableTargetRateExperiment&&);

  bool IsEnabled() const;
  double GetVideoHysteresisFactor() const;
  double GetScreenshareHysteresisFactor() const;

 private:
  FieldTrialParameter<bool> enabled_;
  FieldTrialParameter<double> video_hysteresis_factor_;
  FieldTrialParameter<double> screenshare_hysteresis_factor_;
};

}  // namespace webrtc

#endif  // RTC_BASE_EXPERIMENTS_STABLE_TARGET_RATE_EXPERIMENT_H_
