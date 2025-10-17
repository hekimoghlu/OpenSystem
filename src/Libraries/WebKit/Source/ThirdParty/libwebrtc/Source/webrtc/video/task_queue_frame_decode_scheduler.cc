/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 9, 2021.
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
#include "video/task_queue_frame_decode_scheduler.h"

#include <algorithm>
#include <utility>

#include "api/sequence_checker.h"
#include "api/task_queue/task_queue_base.h"
#include "rtc_base/checks.h"

namespace webrtc {

TaskQueueFrameDecodeScheduler::TaskQueueFrameDecodeScheduler(
    Clock* clock,
    TaskQueueBase* const bookkeeping_queue)
    : clock_(clock), bookkeeping_queue_(bookkeeping_queue) {
  RTC_DCHECK(clock_);
  RTC_DCHECK(bookkeeping_queue_);
}

TaskQueueFrameDecodeScheduler::~TaskQueueFrameDecodeScheduler() {
  RTC_DCHECK(stopped_);
  RTC_DCHECK(!scheduled_rtp_) << "Outstanding scheduled rtp=" << *scheduled_rtp_
                              << ". Call CancelOutstanding before destruction.";
}

void TaskQueueFrameDecodeScheduler::ScheduleFrame(
    uint32_t rtp,
    FrameDecodeTiming::FrameSchedule schedule,
    FrameReleaseCallback cb) {
  RTC_DCHECK(!stopped_) << "Can not schedule frames after stopped.";
  RTC_DCHECK(!scheduled_rtp_.has_value())
      << "Can not schedule two frames for release at the same time.";
  RTC_DCHECK(cb);
  scheduled_rtp_ = rtp;

  TimeDelta wait = std::max(
      TimeDelta::Zero(), schedule.latest_decode_time - clock_->CurrentTime());
  bookkeeping_queue_->PostDelayedHighPrecisionTask(
      SafeTask(task_safety_.flag(),
               [this, rtp, schedule, cb = std::move(cb)]() mutable {
                 RTC_DCHECK_RUN_ON(bookkeeping_queue_);
                 // If the next frame rtp has changed since this task was
                 // this scheduled release should be skipped.
                 if (scheduled_rtp_ != rtp)
                   return;
                 scheduled_rtp_ = std::nullopt;
                 std::move(cb)(rtp, schedule.render_time);
               }),
      wait);
}

void TaskQueueFrameDecodeScheduler::CancelOutstanding() {
  scheduled_rtp_ = std::nullopt;
}

std::optional<uint32_t> TaskQueueFrameDecodeScheduler::ScheduledRtpTimestamp() {
  return scheduled_rtp_;
}

void TaskQueueFrameDecodeScheduler::Stop() {
  CancelOutstanding();
  stopped_ = true;
}

}  // namespace webrtc
