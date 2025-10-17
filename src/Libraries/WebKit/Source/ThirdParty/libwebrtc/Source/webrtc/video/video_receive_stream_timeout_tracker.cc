/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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
#include "video/video_receive_stream_timeout_tracker.h"

#include <algorithm>
#include <utility>

#include "rtc_base/logging.h"

namespace webrtc {

VideoReceiveStreamTimeoutTracker::VideoReceiveStreamTimeoutTracker(
    Clock* clock,
    TaskQueueBase* const bookkeeping_queue,
    const Timeouts& timeouts,
    TimeoutCallback callback)
    : clock_(clock),
      bookkeeping_queue_(bookkeeping_queue),
      timeouts_(timeouts),
      timeout_cb_(std::move(callback)) {}

VideoReceiveStreamTimeoutTracker::~VideoReceiveStreamTimeoutTracker() {
  RTC_DCHECK(!timeout_task_.Running());
}

bool VideoReceiveStreamTimeoutTracker::Running() const {
  return timeout_task_.Running();
}

TimeDelta VideoReceiveStreamTimeoutTracker::TimeUntilTimeout() const {
  return std::max(timeout_ - clock_->CurrentTime(), TimeDelta::Zero());
}

void VideoReceiveStreamTimeoutTracker::Start(bool waiting_for_keyframe) {
  RTC_DCHECK_RUN_ON(bookkeeping_queue_);
  RTC_DCHECK(!timeout_task_.Running());
  waiting_for_keyframe_ = waiting_for_keyframe;
  TimeDelta timeout_delay = TimeoutForNextFrame();
  last_frame_ = clock_->CurrentTime();
  timeout_ = last_frame_ + timeout_delay;
  timeout_task_ =
      RepeatingTaskHandle::DelayedStart(bookkeeping_queue_, timeout_delay,
                                        [this] { return HandleTimeoutTask(); });
}

void VideoReceiveStreamTimeoutTracker::Stop() {
  timeout_task_.Stop();
}

void VideoReceiveStreamTimeoutTracker::SetWaitingForKeyframe() {
  RTC_DCHECK_RUN_ON(bookkeeping_queue_);
  waiting_for_keyframe_ = true;
  TimeDelta timeout_delay = TimeoutForNextFrame();
  if (clock_->CurrentTime() + timeout_delay < timeout_) {
    Stop();
    Start(waiting_for_keyframe_);
  }
}

void VideoReceiveStreamTimeoutTracker::OnEncodedFrameReleased() {
  RTC_DCHECK_RUN_ON(bookkeeping_queue_);
  // If we were waiting for a keyframe, then it has just been released.
  waiting_for_keyframe_ = false;
  last_frame_ = clock_->CurrentTime();
  timeout_ = last_frame_ + TimeoutForNextFrame();
}

TimeDelta VideoReceiveStreamTimeoutTracker::HandleTimeoutTask() {
  RTC_DCHECK_RUN_ON(bookkeeping_queue_);
  Timestamp now = clock_->CurrentTime();
  // `timeout_` is hit and we have timed out. Schedule the next timeout at
  // the timeout delay.
  if (now >= timeout_) {
    RTC_DLOG(LS_VERBOSE) << "Stream timeout at " << now;
    TimeDelta timeout_delay = TimeoutForNextFrame();
    timeout_ = now + timeout_delay;
    timeout_cb_(now - last_frame_);
    return timeout_delay;
  }
  // Otherwise, `timeout_` changed since we scheduled a timeout. Reschedule
  // a timeout check.
  return timeout_ - now;
}

void VideoReceiveStreamTimeoutTracker::SetTimeouts(Timeouts timeouts) {
  RTC_DCHECK_RUN_ON(bookkeeping_queue_);
  timeouts_ = timeouts;
}

}  // namespace webrtc
