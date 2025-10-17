/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
#ifndef MODULES_REMOTE_BITRATE_ESTIMATOR_OVERUSE_ESTIMATOR_H_
#define MODULES_REMOTE_BITRATE_ESTIMATOR_OVERUSE_ESTIMATOR_H_

#include <stdint.h>

#include <deque>

#include "api/transport/bandwidth_usage.h"

namespace webrtc {

class OveruseEstimator {
 public:
  OveruseEstimator();

  OveruseEstimator(const OveruseEstimator&) = delete;
  OveruseEstimator& operator=(const OveruseEstimator&) = delete;

  ~OveruseEstimator() = default;

  // Update the estimator with a new sample. The deltas should represent deltas
  // between timestamp groups as defined by the InterArrival class.
  // `current_hypothesis` should be the hypothesis of the over-use detector at
  // this time.
  void Update(int64_t t_delta,
              double ts_delta,
              int size_delta,
              BandwidthUsage current_hypothesis,
              int64_t now_ms);

  // Returns the estimated noise/jitter variance in ms^2.
  double var_noise() const { return var_noise_; }

  // Returns the estimated inter-arrival time delta offset in ms.
  double offset() const { return offset_; }

  // Returns the number of deltas which the current over-use estimator state is
  // based on.
  int num_of_deltas() const { return num_of_deltas_; }

 private:
  double UpdateMinFramePeriod(double ts_delta);
  void UpdateNoiseEstimate(double residual, double ts_delta, bool stable_state);

  int num_of_deltas_ = 0;
  double slope_ = 8.0 / 512.0;
  double offset_ = 0;
  double prev_offset_ = 0;
  double E_[2][2] = {{100.0, 0.0}, {0.0, 1e-1}};
  double process_noise_[2] = {1e-13, 1e-3};
  double avg_noise_ = 0.0;
  double var_noise_ = 50.0;
  std::deque<double> ts_delta_hist_;
};
}  // namespace webrtc

#endif  // MODULES_REMOTE_BITRATE_ESTIMATOR_OVERUSE_ESTIMATOR_H_
