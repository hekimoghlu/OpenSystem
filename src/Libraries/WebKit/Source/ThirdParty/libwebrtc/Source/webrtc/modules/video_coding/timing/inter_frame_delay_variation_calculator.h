/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
#ifndef MODULES_VIDEO_CODING_TIMING_INTER_FRAME_DELAY_VARIATION_CALCULATOR_H_
#define MODULES_VIDEO_CODING_TIMING_INTER_FRAME_DELAY_VARIATION_CALCULATOR_H_

#include <stdint.h>

#include <optional>

#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/numerics/sequence_number_unwrapper.h"

namespace webrtc {

// This class calculates the inter-frame delay variation (see RFC5481) between
// the current frame (as supplied through the current call to `Calculate`) and
// the previous frame (as supplied through the previous call to `Calculate`).
class InterFrameDelayVariationCalculator {
 public:
  InterFrameDelayVariationCalculator();

  // Resets the calculator.
  void Reset();

  // Calculates the inter-frame delay variation of a frame with the given
  // RTP timestamp. This method is called when the frame is complete.
  std::optional<TimeDelta> Calculate(uint32_t rtp_timestamp, Timestamp now);

 private:
  // The previous wall clock timestamp used in the calculation.
  std::optional<Timestamp> prev_wall_clock_;
  // The previous RTP timestamp used in the calculation.
  int64_t prev_rtp_timestamp_unwrapped_;

  RtpTimestampUnwrapper unwrapper_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_TIMING_INTER_FRAME_DELAY_VARIATION_CALCULATOR_H_
