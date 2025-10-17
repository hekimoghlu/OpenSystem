/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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
#ifndef MODULES_CONGESTION_CONTROLLER_GOOG_CC_BITRATE_ESTIMATOR_H_
#define MODULES_CONGESTION_CONTROLLER_GOOG_CC_BITRATE_ESTIMATOR_H_

#include <stdint.h>

#include <optional>

#include "api/field_trials_view.h"
#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/timestamp.h"
#include "rtc_base/experiments/field_trial_parser.h"

namespace webrtc {

// Computes a bayesian estimate of the throughput given acks containing
// the arrival time and payload size. Samples which are far from the current
// estimate or are based on few packets are given a smaller weight, as they
// are considered to be more likely to have been caused by, e.g., delay spikes
// unrelated to congestion.
class BitrateEstimator {
 public:
  explicit BitrateEstimator(const FieldTrialsView* key_value_config);
  virtual ~BitrateEstimator();
  virtual void Update(Timestamp at_time, DataSize amount, bool in_alr);

  virtual std::optional<DataRate> bitrate() const;
  std::optional<DataRate> PeekRate() const;

  virtual void ExpectFastRateChange();

 private:
  float UpdateWindow(int64_t now_ms,
                     int bytes,
                     int rate_window_ms,
                     bool* is_small_sample);
  int sum_;
  FieldTrialConstrained<int> initial_window_ms_;
  FieldTrialConstrained<int> noninitial_window_ms_;
  FieldTrialParameter<double> uncertainty_scale_;
  FieldTrialParameter<double> uncertainty_scale_in_alr_;
  FieldTrialParameter<double> small_sample_uncertainty_scale_;
  FieldTrialParameter<DataSize> small_sample_threshold_;
  FieldTrialParameter<DataRate> uncertainty_symmetry_cap_;
  FieldTrialParameter<DataRate> estimate_floor_;
  int64_t current_window_ms_;
  int64_t prev_time_ms_;
  float bitrate_estimate_kbps_;
  float bitrate_estimate_var_;
};

}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_GOOG_CC_BITRATE_ESTIMATOR_H_
