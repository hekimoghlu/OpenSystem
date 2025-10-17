/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
#ifndef TEST_PC_E2E_TEST_ACTIVITIES_EXECUTOR_H_
#define TEST_PC_E2E_TEST_ACTIVITIES_EXECUTOR_H_

#include <optional>
#include <queue>
#include <vector>

#include "api/task_queue/task_queue_base.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/task_queue_for_test.h"
#include "rtc_base/task_utils/repeating_task.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {
namespace webrtc_pc_e2e {

class TestActivitiesExecutor {
 public:
  explicit TestActivitiesExecutor(Clock* clock) : clock_(clock) {}
  ~TestActivitiesExecutor() { Stop(); }

  // Starts scheduled activities according to their schedule. All activities
  // that will be scheduled after Start(...) was invoked will be executed
  // immediately according to their schedule.
  void Start(TaskQueueForTest* task_queue) { Start(task_queue->Get()); }
  void Start(TaskQueueBase* task_queue);
  void Stop();

  // Schedule activity to be executed. If test isn't started yet, then activity
  // will be executed according to its schedule after Start() will be invoked.
  // If test is started, then it will be executed immediately according to its
  // schedule.
  void ScheduleActivity(TimeDelta initial_delay_since_start,
                        std::optional<TimeDelta> interval,
                        std::function<void(TimeDelta)> func);

 private:
  struct ScheduledActivity {
    ScheduledActivity(TimeDelta initial_delay_since_start,
                      std::optional<TimeDelta> interval,
                      std::function<void(TimeDelta)> func);

    TimeDelta initial_delay_since_start;
    std::optional<TimeDelta> interval;
    std::function<void(TimeDelta)> func;
  };

  void PostActivity(ScheduledActivity activity)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(lock_);
  Timestamp Now() const;

  Clock* const clock_;

  TaskQueueBase* task_queue_;

  Mutex lock_;
  // Time when test was started. Minus infinity means that it wasn't started
  // yet.
  Timestamp start_time_ RTC_GUARDED_BY(lock_) = Timestamp::MinusInfinity();
  // Queue of activities that were added before test was started.
  // Activities from this queue will be posted on the `task_queue_` after test
  // will be set up and then this queue will be unused.
  std::queue<ScheduledActivity> scheduled_activities_ RTC_GUARDED_BY(lock_);
  // List of task handles for activities, that are posted on `task_queue_` as
  // repeated during the call.
  std::vector<RepeatingTaskHandle> repeating_task_handles_
      RTC_GUARDED_BY(lock_);
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_TEST_ACTIVITIES_EXECUTOR_H_
