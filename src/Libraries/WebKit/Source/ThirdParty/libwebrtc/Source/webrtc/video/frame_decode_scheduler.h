/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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
#ifndef VIDEO_FRAME_DECODE_SCHEDULER_H_
#define VIDEO_FRAME_DECODE_SCHEDULER_H_

#include <stdint.h>

#include <optional>

#include "absl/functional/any_invocable.h"
#include "api/units/timestamp.h"
#include "video/frame_decode_timing.h"

namespace webrtc {

class FrameDecodeScheduler {
 public:
  // Invoked when a frame with `rtp_timestamp` is ready for decoding.
  using FrameReleaseCallback =
      absl::AnyInvocable<void(uint32_t rtp_timestamp,
                              Timestamp render_time) &&>;

  virtual ~FrameDecodeScheduler() = default;

  // Returns the rtp timestamp of the next frame scheduled for release, or
  // `nullopt` if no frame is currently scheduled.
  virtual std::optional<uint32_t> ScheduledRtpTimestamp() = 0;

  // Schedules a frame for release based on `schedule`. When released,
  // `callback` will be invoked with the `rtp` timestamp of the frame and the
  // `render_time`
  virtual void ScheduleFrame(uint32_t rtp,
                             FrameDecodeTiming::FrameSchedule schedule,
                             FrameReleaseCallback callback) = 0;

  // Cancels all scheduled frames.
  virtual void CancelOutstanding() = 0;

  // Stop() Must be called before destruction.
  virtual void Stop() = 0;
};

}  // namespace webrtc

#endif  // VIDEO_FRAME_DECODE_SCHEDULER_H_
