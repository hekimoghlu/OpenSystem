/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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
#ifndef MODULES_CONGESTION_CONTROLLER_GOOG_CC_LINK_CAPACITY_ESTIMATOR_H_
#define MODULES_CONGESTION_CONTROLLER_GOOG_CC_LINK_CAPACITY_ESTIMATOR_H_

#include <optional>

#include "api/units/data_rate.h"

namespace webrtc {
class LinkCapacityEstimator {
 public:
  LinkCapacityEstimator();
  DataRate UpperBound() const;
  DataRate LowerBound() const;
  void Reset();
  void OnOveruseDetected(DataRate acknowledged_rate);
  void OnProbeRate(DataRate probe_rate);
  bool has_estimate() const;
  DataRate estimate() const;

 private:
  friend class GoogCcStatePrinter;
  void Update(DataRate capacity_sample, double alpha);

  double deviation_estimate_kbps() const;
  std::optional<double> estimate_kbps_;
  double deviation_kbps_ = 0.4;
};
}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_GOOG_CC_LINK_CAPACITY_ESTIMATOR_H_
