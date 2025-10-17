/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#ifndef RTC_BASE_EXPERIMENTS_ALR_EXPERIMENT_H_
#define RTC_BASE_EXPERIMENTS_ALR_EXPERIMENT_H_

#include <stdint.h>

#include <optional>

#include "absl/strings/string_view.h"
#include "api/field_trials_view.h"

namespace webrtc {
struct AlrExperimentSettings {
 public:
  float pacing_factor;
  int64_t max_paced_queue_time;
  int alr_bandwidth_usage_percent;
  int alr_start_budget_level_percent;
  int alr_stop_budget_level_percent;
  // Will be sent to the receive side for stats slicing.
  // Can be 0..6, because it's sent as a 3 bits value and there's also
  // reserved value to indicate absence of experiment.
  int group_id;

  static constexpr absl::string_view kScreenshareProbingBweExperimentName =
      "WebRTC-ProbingScreenshareBwe";
  static constexpr absl::string_view kStrictPacingAndProbingExperimentName =
      "WebRTC-StrictPacingAndProbing";

  static std::optional<AlrExperimentSettings> CreateFromFieldTrial(
      const FieldTrialsView& key_value_config,
      absl::string_view experiment_name);
  static bool MaxOneFieldTrialEnabled(const FieldTrialsView& key_value_config);

 private:
  AlrExperimentSettings() = default;
};
}  // namespace webrtc

#endif  // RTC_BASE_EXPERIMENTS_ALR_EXPERIMENT_H_
