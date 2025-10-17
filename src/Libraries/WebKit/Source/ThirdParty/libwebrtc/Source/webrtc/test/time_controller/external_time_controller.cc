/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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
#include "test/time_controller/external_time_controller.h"

#include <algorithm>
#include <map>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "api/task_queue/task_queue_base.h"
#include "api/task_queue/task_queue_factory.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/checks.h"
#include "rtc_base/synchronization/yield_policy.h"
#include "test/time_controller/simulated_time_controller.h"

namespace webrtc {

// Wraps a TaskQueue so that it can reschedule the time controller whenever
// an external call schedules a new task.
class ExternalTimeController::TaskQueueWrapper : public TaskQueueBase {
 public:
  TaskQueueWrapper(ExternalTimeController* parent,
                   std::unique_ptr<TaskQueueBase, TaskQueueDeleter> base)
      : parent_(parent), base_(std::move(base)) {}

  void PostTaskImpl(absl::AnyInvocable<void() &&> task,
                    const PostTaskTraits& traits,
                    const Location& location) override {
    parent_->UpdateTime();
    base_->PostTask(TaskWrapper(std::move(task)));
    parent_->ScheduleNext();
  }

  void PostDelayedTaskImpl(absl::AnyInvocable<void() &&> task,
                           TimeDelta delay,
                           const PostDelayedTaskTraits& traits,
                           const Location& location) override {
    parent_->UpdateTime();
    if (traits.high_precision) {
      base_->PostDelayedHighPrecisionTask(TaskWrapper(std::move(task)), delay);
    } else {
      base_->PostDelayedTask(TaskWrapper(std::move(task)), delay);
    }
    parent_->ScheduleNext();
  }

  void Delete() override { delete this; }

 private:
  absl::AnyInvocable<void() &&> TaskWrapper(
      absl::AnyInvocable<void() &&> task) {
    return [task = std::move(task), this]() mutable {
      CurrentTaskQueueSetter current(this);
      std::move(task)();
    };
  }

  ExternalTimeController* const parent_;
  std::unique_ptr<TaskQueueBase, TaskQueueDeleter> base_;
};

ExternalTimeController::ExternalTimeController(ControlledAlarmClock* alarm)
    : alarm_(alarm),
      impl_(alarm_->GetClock()->CurrentTime()),
      yield_policy_(&impl_) {
  global_clock_.SetTime(alarm_->GetClock()->CurrentTime());
  alarm_->SetCallback([this] { Run(); });
}

Clock* ExternalTimeController::GetClock() {
  return alarm_->GetClock();
}

TaskQueueFactory* ExternalTimeController::GetTaskQueueFactory() {
  return this;
}

void ExternalTimeController::AdvanceTime(TimeDelta duration) {
  alarm_->Sleep(duration);
}

std::unique_ptr<rtc::Thread> ExternalTimeController::CreateThread(
    const std::string& name,
    std::unique_ptr<rtc::SocketServer> socket_server) {
  RTC_DCHECK_NOTREACHED();
  return nullptr;
}

rtc::Thread* ExternalTimeController::GetMainThread() {
  RTC_DCHECK_NOTREACHED();
  return nullptr;
}

std::unique_ptr<TaskQueueBase, TaskQueueDeleter>
ExternalTimeController::CreateTaskQueue(
    absl::string_view name,
    TaskQueueFactory::Priority priority) const {
  return std::unique_ptr<TaskQueueBase, TaskQueueDeleter>(
      new TaskQueueWrapper(const_cast<ExternalTimeController*>(this),
                           impl_.CreateTaskQueue(name, priority)));
}

void ExternalTimeController::Run() {
  rtc::ScopedYieldPolicy yield_policy(&impl_);
  UpdateTime();
  impl_.RunReadyRunners();
  ScheduleNext();
}

void ExternalTimeController::UpdateTime() {
  Timestamp now = alarm_->GetClock()->CurrentTime();
  impl_.AdvanceTime(now);
  global_clock_.SetTime(now);
}

void ExternalTimeController::ScheduleNext() {
  RTC_DCHECK_EQ(impl_.CurrentTime(), alarm_->GetClock()->CurrentTime());
  TimeDelta delay =
      std::max(impl_.NextRunTime() - impl_.CurrentTime(), TimeDelta::Zero());
  if (delay.IsFinite()) {
    alarm_->ScheduleAlarmAt(alarm_->GetClock()->CurrentTime() + delay);
  }
}

}  // namespace webrtc
