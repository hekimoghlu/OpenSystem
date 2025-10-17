/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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
#ifndef TEST_TIME_CONTROLLER_EXTERNAL_TIME_CONTROLLER_H_
#define TEST_TIME_CONTROLLER_EXTERNAL_TIME_CONTROLLER_H_

#include <functional>
#include <memory>

#include "absl/strings/string_view.h"
#include "api/task_queue/task_queue_base.h"
#include "api/task_queue/task_queue_factory.h"
#include "api/test/time_controller.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "system_wrappers/include/clock.h"
#include "test/time_controller/simulated_time_controller.h"

namespace webrtc {

// TimeController implementation built on an external controlled alarm.
// This implementation is used to delegate scheduling and execution to an
// external run loop.
class ExternalTimeController : public TimeController, public TaskQueueFactory {
 public:
  explicit ExternalTimeController(ControlledAlarmClock* alarm);

  // Implementation of TimeController.
  Clock* GetClock() override;
  TaskQueueFactory* GetTaskQueueFactory() override;
  void AdvanceTime(TimeDelta duration) override;
  std::unique_ptr<rtc::Thread> CreateThread(
      const std::string& name,
      std::unique_ptr<rtc::SocketServer> socket_server) override;
  rtc::Thread* GetMainThread() override;

  // Implementation of TaskQueueFactory.
  std::unique_ptr<TaskQueueBase, TaskQueueDeleter> CreateTaskQueue(
      absl::string_view name,
      TaskQueueFactory::Priority priority) const override;

 private:
  class TaskQueueWrapper;

  // Executes any tasks scheduled at or before the current time.  May call
  // `ScheduleNext` to schedule the next call to `Run`.
  void Run();

  void UpdateTime();
  void ScheduleNext();

  ControlledAlarmClock* alarm_;
  sim_time_impl::SimulatedTimeControllerImpl impl_;
  rtc::ScopedYieldPolicy yield_policy_;

  // Overrides the global rtc::Clock to ensure that it reports the same times as
  // the time controller.
  rtc::ScopedBaseFakeClock global_clock_;
};

}  // namespace webrtc

#endif  // TEST_TIME_CONTROLLER_EXTERNAL_TIME_CONTROLLER_H_
