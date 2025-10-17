/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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
#include "video/frame_decode_timing.h"

#include <algorithm>
#include <optional>

#include "api/units/time_delta.h"
#include "rtc_base/logging.h"

namespace webrtc {

FrameDecodeTiming::FrameDecodeTiming(Clock* clock,
                                     webrtc::VCMTiming const* timing)
    : clock_(clock), timing_(timing) {
  RTC_DCHECK(clock_);
  RTC_DCHECK(timing_);
}

std::optional<FrameDecodeTiming::FrameSchedule>
FrameDecodeTiming::OnFrameBufferUpdated(uint32_t next_temporal_unit_rtp,
                                        uint32_t last_temporal_unit_rtp,
                                        TimeDelta max_wait_for_frame,
                                        bool too_many_frames_queued) {
  RTC_DCHECK_GE(max_wait_for_frame, TimeDelta::Zero());
  const Timestamp now = clock_->CurrentTime();
  Timestamp render_time = timing_->RenderTime(next_temporal_unit_rtp, now);
  TimeDelta max_wait =
      timing_->MaxWaitingTime(render_time, now, too_many_frames_queued);

  // If the delay is not too far in the past, or this is the last decodable
  // frame then it is the best frame to be decoded. Otherwise, fast-forward
  // to the next frame in the buffer.
  if (max_wait <= -kMaxAllowedFrameDelay &&
      next_temporal_unit_rtp != last_temporal_unit_rtp) {
    RTC_DLOG(LS_VERBOSE) << "Fast-forwarded frame " << next_temporal_unit_rtp
                         << " render time " << render_time << " with delay "
                         << max_wait;
    return std::nullopt;
  }

  max_wait.Clamp(TimeDelta::Zero(), max_wait_for_frame);
  RTC_DLOG(LS_VERBOSE) << "Selected frame with rtp " << next_temporal_unit_rtp
                       << " render time " << render_time
                       << " with a max wait of " << max_wait_for_frame
                       << " clamped to " << max_wait;
  Timestamp latest_decode_time = now + max_wait;
  return FrameSchedule{.latest_decode_time = latest_decode_time,
                       .render_time = render_time};
}

}  // namespace webrtc
