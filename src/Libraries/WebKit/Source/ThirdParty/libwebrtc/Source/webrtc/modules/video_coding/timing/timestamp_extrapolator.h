/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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
#ifndef MODULES_VIDEO_CODING_TIMING_TIMESTAMP_EXTRAPOLATOR_H_
#define MODULES_VIDEO_CODING_TIMING_TIMESTAMP_EXTRAPOLATOR_H_

#include <stdint.h>

#include <optional>

#include "api/units/timestamp.h"
#include "rtc_base/numerics/sequence_number_unwrapper.h"

namespace webrtc {

// Not thread safe.
class TimestampExtrapolator {
 public:
  explicit TimestampExtrapolator(Timestamp start);
  void Update(Timestamp now, uint32_t ts90khz);
  std::optional<Timestamp> ExtrapolateLocalTime(uint32_t timestamp90khz) const;
  void Reset(Timestamp start);

 private:
  void CheckForWrapArounds(uint32_t ts90khz);
  bool DelayChangeDetection(double error);

  double w_[2];
  double p_[2][2];
  Timestamp start_;
  Timestamp prev_;
  std::optional<int64_t> first_unwrapped_timestamp_;
  RtpTimestampUnwrapper unwrapper_;
  std::optional<int64_t> prev_unwrapped_timestamp_;
  uint32_t packet_count_;
  double detector_accumulator_pos_;
  double detector_accumulator_neg_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_TIMING_TIMESTAMP_EXTRAPOLATOR_H_
