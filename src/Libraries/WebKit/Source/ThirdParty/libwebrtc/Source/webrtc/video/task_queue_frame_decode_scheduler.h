/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#ifndef VIDEO_TASK_QUEUE_FRAME_DECODE_SCHEDULER_H_
#define VIDEO_TASK_QUEUE_FRAME_DECODE_SCHEDULER_H_

#include "video/frame_decode_scheduler.h"

namespace webrtc {

// An implementation of FrameDecodeScheduler that is based on TaskQueues. This
// is the default implementation for general use.
class TaskQueueFrameDecodeScheduler : public FrameDecodeScheduler {
 public:
  TaskQueueFrameDecodeScheduler(Clock* clock,
                                TaskQueueBase* const bookkeeping_queue);
  ~TaskQueueFrameDecodeScheduler() override;
  TaskQueueFrameDecodeScheduler(const TaskQueueFrameDecodeScheduler&) = delete;
  TaskQueueFrameDecodeScheduler& operator=(
      const TaskQueueFrameDecodeScheduler&) = delete;

  // FrameDecodeScheduler implementation.
  std::optional<uint32_t> ScheduledRtpTimestamp() override;
  void ScheduleFrame(uint32_t rtp,
                     FrameDecodeTiming::FrameSchedule schedule,
                     FrameReleaseCallback cb) override;
  void CancelOutstanding() override;
  void Stop() override;

 private:
  Clock* const clock_;
  TaskQueueBase* const bookkeeping_queue_;

  std::optional<uint32_t> scheduled_rtp_;
  ScopedTaskSafetyDetached task_safety_;
  bool stopped_ = false;
};

}  // namespace webrtc

#endif  // VIDEO_TASK_QUEUE_FRAME_DECODE_SCHEDULER_H_
