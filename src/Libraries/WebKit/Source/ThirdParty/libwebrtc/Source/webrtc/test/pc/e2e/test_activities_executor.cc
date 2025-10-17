/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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
#include "test/pc/e2e/test_activities_executor.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/task_queue_for_test.h"

namespace webrtc {
namespace webrtc_pc_e2e {

void TestActivitiesExecutor::Start(TaskQueueBase* task_queue) {
  RTC_DCHECK(task_queue);
  task_queue_ = task_queue;
  MutexLock lock(&lock_);
  start_time_ = Now();
  while (!scheduled_activities_.empty()) {
    PostActivity(std::move(scheduled_activities_.front()));
    scheduled_activities_.pop();
  }
}

void TestActivitiesExecutor::Stop() {
  if (task_queue_ == nullptr) {
    // Already stopped or not started.
    return;
  }
  SendTask(task_queue_, [this]() {
    MutexLock lock(&lock_);
    for (auto& handle : repeating_task_handles_) {
      handle.Stop();
    }
  });
  task_queue_ = nullptr;
}

void TestActivitiesExecutor::ScheduleActivity(
    TimeDelta initial_delay_since_start,
    std::optional<TimeDelta> interval,
    std::function<void(TimeDelta)> func) {
  RTC_CHECK(initial_delay_since_start.IsFinite() &&
            initial_delay_since_start >= TimeDelta::Zero());
  RTC_CHECK(!interval ||
            (interval->IsFinite() && *interval > TimeDelta::Zero()));
  MutexLock lock(&lock_);
  ScheduledActivity activity(initial_delay_since_start, interval, func);
  if (start_time_.IsInfinite()) {
    scheduled_activities_.push(std::move(activity));
  } else {
    PostActivity(std::move(activity));
  }
}

void TestActivitiesExecutor::PostActivity(ScheduledActivity activity) {
  // Because start_time_ will never change at this point copy it to local
  // variable to capture in in lambda without requirement to hold a lock.
  Timestamp start_time = start_time_;

  TimeDelta remaining_delay =
      activity.initial_delay_since_start == TimeDelta::Zero()
          ? TimeDelta::Zero()
          : activity.initial_delay_since_start - (Now() - start_time);
  if (remaining_delay < TimeDelta::Zero()) {
    RTC_LOG(LS_WARNING) << "Executing late task immediately, late by="
                        << ToString(remaining_delay.Abs());
    remaining_delay = TimeDelta::Zero();
  }

  if (activity.interval) {
    if (remaining_delay == TimeDelta::Zero()) {
      repeating_task_handles_.push_back(RepeatingTaskHandle::Start(
          task_queue_, [activity, start_time, this]() {
            activity.func(Now() - start_time);
            return *activity.interval;
          }));
      return;
    }
    repeating_task_handles_.push_back(RepeatingTaskHandle::DelayedStart(
        task_queue_, remaining_delay, [activity, start_time, this]() {
          activity.func(Now() - start_time);
          return *activity.interval;
        }));
    return;
  }

  if (remaining_delay == TimeDelta::Zero()) {
    task_queue_->PostTask(
        [activity, start_time, this]() { activity.func(Now() - start_time); });
    return;
  }

  task_queue_->PostDelayedTask(
      [activity, start_time, this]() { activity.func(Now() - start_time); },
      remaining_delay);
}

Timestamp TestActivitiesExecutor::Now() const {
  return clock_->CurrentTime();
}

TestActivitiesExecutor::ScheduledActivity::ScheduledActivity(
    TimeDelta initial_delay_since_start,
    std::optional<TimeDelta> interval,
    std::function<void(TimeDelta)> func)
    : initial_delay_since_start(initial_delay_since_start),
      interval(interval),
      func(std::move(func)) {}

}  // namespace webrtc_pc_e2e
}  // namespace webrtc
