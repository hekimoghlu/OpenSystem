/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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
#include "modules/congestion_controller/goog_cc/link_capacity_estimator.h"

#include <algorithm>
#include <cmath>

#include "api/units/data_rate.h"
#include "rtc_base/numerics/safe_minmax.h"

namespace webrtc {
LinkCapacityEstimator::LinkCapacityEstimator() {}

DataRate LinkCapacityEstimator::UpperBound() const {
  if (estimate_kbps_.has_value())
    return DataRate::KilobitsPerSec(estimate_kbps_.value() +
                                    3 * deviation_estimate_kbps());
  return DataRate::Infinity();
}

DataRate LinkCapacityEstimator::LowerBound() const {
  if (estimate_kbps_.has_value())
    return DataRate::KilobitsPerSec(
        std::max(0.0, estimate_kbps_.value() - 3 * deviation_estimate_kbps()));
  return DataRate::Zero();
}

void LinkCapacityEstimator::Reset() {
  estimate_kbps_.reset();
}

void LinkCapacityEstimator::OnOveruseDetected(DataRate acknowledged_rate) {
  Update(acknowledged_rate, 0.05);
}

void LinkCapacityEstimator::OnProbeRate(DataRate probe_rate) {
  Update(probe_rate, 0.5);
}

void LinkCapacityEstimator::Update(DataRate capacity_sample, double alpha) {
  double sample_kbps = capacity_sample.kbps();
  if (!estimate_kbps_.has_value()) {
    estimate_kbps_ = sample_kbps;
  } else {
    estimate_kbps_ = (1 - alpha) * estimate_kbps_.value() + alpha * sample_kbps;
  }
  // Estimate the variance of the link capacity estimate and normalize the
  // variance with the link capacity estimate.
  const double norm = std::max(estimate_kbps_.value(), 1.0);
  double error_kbps = estimate_kbps_.value() - sample_kbps;
  deviation_kbps_ =
      (1 - alpha) * deviation_kbps_ + alpha * error_kbps * error_kbps / norm;
  // 0.4 ~= 14 kbit/s at 500 kbit/s
  // 2.5f ~= 35 kbit/s at 500 kbit/s
  deviation_kbps_ = rtc::SafeClamp(deviation_kbps_, 0.4f, 2.5f);
}

bool LinkCapacityEstimator::has_estimate() const {
  return estimate_kbps_.has_value();
}

DataRate LinkCapacityEstimator::estimate() const {
  return DataRate::KilobitsPerSec(*estimate_kbps_);
}

double LinkCapacityEstimator::deviation_estimate_kbps() const {
  // Calculate the max bit rate std dev given the normalized
  // variance and the current throughput bitrate. The standard deviation will
  // only be used if estimate_kbps_ has a value.
  return sqrt(deviation_kbps_ * estimate_kbps_.value());
}
}  // namespace webrtc
