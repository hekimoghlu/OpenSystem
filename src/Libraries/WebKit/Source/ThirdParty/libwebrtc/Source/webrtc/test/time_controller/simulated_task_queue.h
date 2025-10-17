/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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
#ifndef TEST_TIME_CONTROLLER_SIMULATED_TASK_QUEUE_H_
#define TEST_TIME_CONTROLLER_SIMULATED_TASK_QUEUE_H_

#include <deque>
#include <map>
#include <memory>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "api/units/time_delta.h"
#include "rtc_base/synchronization/mutex.h"
#include "test/time_controller/simulated_time_controller.h"

namespace webrtc {

class SimulatedTaskQueue : public TaskQueueBase,
                           public sim_time_impl::SimulatedSequenceRunner {
 public:
  SimulatedTaskQueue(sim_time_impl::SimulatedTimeControllerImpl* handler,
                     absl::string_view name);

  ~SimulatedTaskQueue();

  void RunReady(Timestamp at_time) override;

  Timestamp GetNextRunTime() const override {
    MutexLock lock(&lock_);
    return next_run_time_;
  }
  TaskQueueBase* GetAsTaskQueue() override { return this; }

  // TaskQueueBase interface
  void Delete() override;
  void PostTaskImpl(absl::AnyInvocable<void() &&> task,
                    const PostTaskTraits& traits,
                    const Location& location) override;
  void PostDelayedTaskImpl(absl::AnyInvocable<void() &&> task,
                           TimeDelta delay,
                           const PostDelayedTaskTraits& traits,
                           const Location& location) override;

 private:
  sim_time_impl::SimulatedTimeControllerImpl* const handler_;
  // Using char* to be debugger friendly.
  char* name_;

  mutable Mutex lock_;

  std::deque<absl::AnyInvocable<void() &&>> ready_tasks_ RTC_GUARDED_BY(lock_);
  std::map<Timestamp, std::vector<absl::AnyInvocable<void() &&>>> delayed_tasks_
      RTC_GUARDED_BY(lock_);

  Timestamp next_run_time_ RTC_GUARDED_BY(lock_) = Timestamp::PlusInfinity();
};

}  // namespace webrtc

#endif  // TEST_TIME_CONTROLLER_SIMULATED_TASK_QUEUE_H_
