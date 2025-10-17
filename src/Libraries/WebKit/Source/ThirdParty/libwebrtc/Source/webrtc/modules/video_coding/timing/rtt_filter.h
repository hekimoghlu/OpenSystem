/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
#ifndef MODULES_VIDEO_CODING_TIMING_RTT_FILTER_H_
#define MODULES_VIDEO_CODING_TIMING_RTT_FILTER_H_

#include <stdint.h>

#include "absl/container/inlined_vector.h"
#include "api/units/time_delta.h"

namespace webrtc {

class RttFilter {
 public:
  RttFilter();
  RttFilter(const RttFilter&) = delete;
  RttFilter& operator=(const RttFilter&) = delete;

  // Resets the filter.
  void Reset();
  // Updates the filter with a new sample.
  void Update(TimeDelta rtt);
  // A getter function for the current RTT level.
  TimeDelta Rtt() const;

 private:
  // The size of the drift and jump memory buffers
  // and thus also the detection threshold for these
  // detectors in number of samples.
  static constexpr int kMaxDriftJumpCount = 5;
  using BufferList = absl::InlinedVector<TimeDelta, kMaxDriftJumpCount>;

  // Detects RTT jumps by comparing the difference between
  // samples and average to the standard deviation.
  // Returns true if the long time statistics should be updated
  // and false otherwise
  bool JumpDetection(TimeDelta rtt);

  // Detects RTT drifts by comparing the difference between
  // max and average to the standard deviation.
  // Returns true if the long time statistics should be updated
  // and false otherwise
  bool DriftDetection(TimeDelta rtt);

  // Computes the short time average and maximum of the vector buf.
  void ShortRttFilter(const BufferList& buf);

  bool got_non_zero_update_;
  TimeDelta avg_rtt_;
  // Variance units are TimeDelta^2. Store as ms^2.
  int64_t var_rtt_;
  TimeDelta max_rtt_;
  uint32_t filt_fact_count_;
  bool last_jump_positive_ = false;
  BufferList jump_buf_;
  BufferList drift_buf_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_TIMING_RTT_FILTER_H_
