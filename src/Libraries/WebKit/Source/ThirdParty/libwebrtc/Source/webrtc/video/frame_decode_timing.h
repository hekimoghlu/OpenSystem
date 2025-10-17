/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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
#ifndef VIDEO_FRAME_DECODE_TIMING_H_
#define VIDEO_FRAME_DECODE_TIMING_H_

#include <stdint.h>

#include <functional>

#include "api/task_queue/pending_task_safety_flag.h"
#include "api/task_queue/task_queue_base.h"
#include "modules/video_coding/timing/timing.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

class FrameDecodeTiming {
 public:
  FrameDecodeTiming(Clock* clock, webrtc::VCMTiming const* timing);
  ~FrameDecodeTiming() = default;
  FrameDecodeTiming(const FrameDecodeTiming&) = delete;
  FrameDecodeTiming& operator=(const FrameDecodeTiming&) = delete;

  // Any frame that has decode delay more than this in the past can be
  // fast-forwarded.
  static constexpr TimeDelta kMaxAllowedFrameDelay = TimeDelta::Millis(5);

  struct FrameSchedule {
    Timestamp latest_decode_time;
    Timestamp render_time;
  };

  std::optional<FrameSchedule> OnFrameBufferUpdated(
      uint32_t next_temporal_unit_rtp,
      uint32_t last_temporal_unit_rtp,
      TimeDelta max_wait_for_frame,
      bool too_many_frames_queued);

 private:
  Clock* const clock_;
  webrtc::VCMTiming const* const timing_;
};

}  // namespace webrtc

#endif  // VIDEO_FRAME_DECODE_TIMING_H_
