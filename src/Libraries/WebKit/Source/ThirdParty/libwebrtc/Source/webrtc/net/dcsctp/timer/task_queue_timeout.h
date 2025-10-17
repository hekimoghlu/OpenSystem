/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
#ifndef NET_DCSCTP_TIMER_TASK_QUEUE_TIMEOUT_H_
#define NET_DCSCTP_TIMER_TASK_QUEUE_TIMEOUT_H_

#include <memory>
#include <utility>

#include "api/task_queue/pending_task_safety_flag.h"
#include "api/task_queue/task_queue_base.h"
#include "api/units/timestamp.h"
#include "net/dcsctp/public/timeout.h"

namespace dcsctp {

// The TaskQueueTimeoutFactory creates `Timeout` instances, which schedules
// itself to be triggered on the provided `task_queue`, which may be a thread,
// an actual TaskQueue or something else which supports posting a delayed task.
//
// Note that each `DcSctpSocket` must have its own `TaskQueueTimeoutFactory`,
// as the `TimeoutID` are not unique among sockets.
//
// This class must outlive any created Timeout that it has created. Note that
// the `DcSctpSocket` will ensure that all Timeouts are deleted when the socket
// is destructed, so this means that this class must outlive the `DcSctpSocket`.
//
// This class, and the timeouts created it, are not thread safe.
class TaskQueueTimeoutFactory {
 public:
  // The `get_time` function must return the current time, relative to any
  // epoch. Whenever a timeout expires, the `on_expired` callback will be
  // triggered, and then the client should provided `timeout_id` to
  // `DcSctpSocketInterface::HandleTimeout`.
  TaskQueueTimeoutFactory(webrtc::TaskQueueBase& task_queue,
                          std::function<TimeMs()> get_time,
                          std::function<void(TimeoutID timeout_id)> on_expired)
      : task_queue_(task_queue),
        get_time_(std::move(get_time)),
        on_expired_(std::move(on_expired)) {}

  // Creates an implementation of `Timeout`.
  std::unique_ptr<Timeout> CreateTimeout(
      webrtc::TaskQueueBase::DelayPrecision precision =
          webrtc::TaskQueueBase::DelayPrecision::kLow) {
    return std::make_unique<TaskQueueTimeout>(*this, precision);
  }

 private:
  class TaskQueueTimeout : public Timeout {
   public:
    TaskQueueTimeout(TaskQueueTimeoutFactory& parent,
                     webrtc::TaskQueueBase::DelayPrecision precision);
    ~TaskQueueTimeout();

    void Start(DurationMs duration_ms, TimeoutID timeout_id) override;
    void Stop() override;

   private:
    TaskQueueTimeoutFactory& parent_;
    const webrtc::TaskQueueBase::DelayPrecision precision_;
    // A safety flag to ensure that posted tasks to the task queue don't
    // reference these object when they go out of scope. Note that this safety
    // flag will be re-created if the scheduled-but-not-yet-expired task is not
    // to be run. This happens when there is a posted delayed task with an
    // expiration time _further away_ than what is now the expected expiration
    // time. In this scenario, a new delayed task has to be posted with a
    // shorter duration and the old task has to be forgotten.
    rtc::scoped_refptr<webrtc::PendingTaskSafetyFlag> pending_task_safety_flag_;
    // The time when the posted delayed task is set to expire. Will be set to
    // the infinite future if there is no such task running.
    webrtc::Timestamp posted_task_expiration_ =
        webrtc::Timestamp::PlusInfinity();
    // The time when the timeout expires. It will be set to the infinite future
    // if the timeout is not running/not started.
    webrtc::Timestamp timeout_expiration_ = webrtc::Timestamp::PlusInfinity();
    // The current timeout ID that will be reported when expired.
    TimeoutID timeout_id_ = TimeoutID(0);
  };

  webrtc::Timestamp Now() { return webrtc::Timestamp::Millis(*get_time_()); }

  RTC_NO_UNIQUE_ADDRESS webrtc::SequenceChecker thread_checker_;
  webrtc::TaskQueueBase& task_queue_;
  const std::function<TimeMs()> get_time_;
  const std::function<void(TimeoutID)> on_expired_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_TIMER_TASK_QUEUE_TIMEOUT_H_
