/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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
#ifndef MODULES_REMOTE_BITRATE_ESTIMATOR_OVERUSE_DETECTOR_H_
#define MODULES_REMOTE_BITRATE_ESTIMATOR_OVERUSE_DETECTOR_H_

#include <stdint.h>

#include "api/transport/bandwidth_usage.h"

namespace webrtc {

class OveruseDetector {
 public:
  OveruseDetector();

  OveruseDetector(const OveruseDetector&) = delete;
  OveruseDetector& operator=(const OveruseDetector&) = delete;

  ~OveruseDetector() = default;

  // Update the detection state based on the estimated inter-arrival time delta
  // offset. `timestamp_delta` is the delta between the last timestamp which the
  // estimated offset is based on and the last timestamp on which the last
  // offset was based on, representing the time between detector updates.
  // `num_of_deltas` is the number of deltas the offset estimate is based on.
  // Returns the state after the detection update.
  BandwidthUsage Detect(double offset,
                        double timestamp_delta,
                        int num_of_deltas,
                        int64_t now_ms);

  // Returns the current detector state.
  BandwidthUsage State() const;

 private:
  void UpdateThreshold(double modified_offset, int64_t now_ms);

  double threshold_ = 12.5;
  int64_t last_update_ms_ = -1;
  double prev_offset_ = 0.0;
  double time_over_using_ = -1;
  int overuse_counter_ = 0;
  BandwidthUsage hypothesis_ = BandwidthUsage::kBwNormal;
};
}  // namespace webrtc

#endif  // MODULES_REMOTE_BITRATE_ESTIMATOR_OVERUSE_DETECTOR_H_
