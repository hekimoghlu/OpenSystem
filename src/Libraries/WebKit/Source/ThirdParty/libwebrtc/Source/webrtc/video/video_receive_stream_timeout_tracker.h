/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#ifndef VIDEO_VIDEO_RECEIVE_STREAM_TIMEOUT_TRACKER_H_
#define VIDEO_VIDEO_RECEIVE_STREAM_TIMEOUT_TRACKER_H_

#include <functional>

#include "api/task_queue/task_queue_base.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/task_utils/repeating_task.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

class VideoReceiveStreamTimeoutTracker {
 public:
  struct Timeouts {
    TimeDelta max_wait_for_keyframe;
    TimeDelta max_wait_for_frame;
  };

  using TimeoutCallback = std::function<void(TimeDelta wait)>;
  VideoReceiveStreamTimeoutTracker(Clock* clock,
                                   TaskQueueBase* const bookkeeping_queue,
                                   const Timeouts& timeouts,
                                   TimeoutCallback callback);
  ~VideoReceiveStreamTimeoutTracker();
  VideoReceiveStreamTimeoutTracker(const VideoReceiveStreamTimeoutTracker&) =
      delete;
  VideoReceiveStreamTimeoutTracker& operator=(
      const VideoReceiveStreamTimeoutTracker&) = delete;

  bool Running() const;
  void Start(bool waiting_for_keyframe);
  void Stop();
  void SetWaitingForKeyframe();
  void OnEncodedFrameReleased();
  TimeDelta TimeUntilTimeout() const;

  void SetTimeouts(Timeouts timeouts);

 private:
  TimeDelta TimeoutForNextFrame() const RTC_RUN_ON(bookkeeping_queue_) {
    return waiting_for_keyframe_ ? timeouts_.max_wait_for_keyframe
                                 : timeouts_.max_wait_for_frame;
  }
  TimeDelta HandleTimeoutTask();

  Clock* const clock_;
  TaskQueueBase* const bookkeeping_queue_;
  Timeouts timeouts_ RTC_GUARDED_BY(bookkeeping_queue_);
  const TimeoutCallback timeout_cb_;
  RepeatingTaskHandle timeout_task_;

  Timestamp last_frame_ = Timestamp::MinusInfinity();
  Timestamp timeout_ = Timestamp::MinusInfinity();
  bool waiting_for_keyframe_;
};
}  // namespace webrtc

#endif  // VIDEO_VIDEO_RECEIVE_STREAM_TIMEOUT_TRACKER_H_
