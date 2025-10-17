/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#include "modules/video_coding/timing/inter_frame_delay_variation_calculator.h"

#include <optional>

#include "api/units/frequency.h"
#include "api/units/time_delta.h"
#include "modules/include/module_common_types_public.h"

namespace webrtc {

namespace {
constexpr Frequency k90kHz = Frequency::KiloHertz(90);
}

InterFrameDelayVariationCalculator::InterFrameDelayVariationCalculator() {
  Reset();
}

void InterFrameDelayVariationCalculator::Reset() {
  prev_wall_clock_ = std::nullopt;
  prev_rtp_timestamp_unwrapped_ = 0;
}

std::optional<TimeDelta> InterFrameDelayVariationCalculator::Calculate(
    uint32_t rtp_timestamp,
    Timestamp now) {
  int64_t rtp_timestamp_unwrapped = unwrapper_.Unwrap(rtp_timestamp);

  if (!prev_wall_clock_) {
    prev_wall_clock_ = now;
    prev_rtp_timestamp_unwrapped_ = rtp_timestamp_unwrapped;
    // Inter-frame delay variation is undefined for a single frame.
    // TODO(brandtr): Should this return std::nullopt instead?
    return TimeDelta::Zero();
  }

  // Account for reordering in jitter variance estimate in the future?
  // Note that this also captures incomplete frames which are grabbed for
  // decoding after a later frame has been complete, i.e. real packet losses.
  uint32_t cropped_prev = static_cast<uint32_t>(prev_rtp_timestamp_unwrapped_);
  if (rtp_timestamp_unwrapped < prev_rtp_timestamp_unwrapped_ ||
      !IsNewerTimestamp(rtp_timestamp, cropped_prev)) {
    return std::nullopt;
  }

  // Compute the compensated timestamp difference.
  TimeDelta delta_wall = now - *prev_wall_clock_;
  int64_t d_rtp_ticks = rtp_timestamp_unwrapped - prev_rtp_timestamp_unwrapped_;
  TimeDelta delta_rtp = d_rtp_ticks / k90kHz;

  // The inter-frame delay variation is the second order difference between the
  // RTP and wall clocks of the two frames, or in other words, the first order
  // difference between `delta_rtp` and `delta_wall`.
  TimeDelta inter_frame_delay_variation = delta_wall - delta_rtp;

  prev_wall_clock_ = now;
  prev_rtp_timestamp_unwrapped_ = rtp_timestamp_unwrapped;

  return inter_frame_delay_variation;
}

}  // namespace webrtc
