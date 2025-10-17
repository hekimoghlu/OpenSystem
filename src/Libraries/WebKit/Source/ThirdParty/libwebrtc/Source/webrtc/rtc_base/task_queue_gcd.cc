/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
// This file contains the implementation of TaskQueue for Mac and iOS.
// The implementation uses Grand Central Dispatch queues (GCD) to
// do the actual task queuing.

#include "rtc_base/task_queue_gcd.h"

#include <dispatch/dispatch.h>
#include <string.h>

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "api/location.h"
#include "api/task_queue/task_queue_base.h"
#include "api/units/time_delta.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/system/gcd_helpers.h"

namespace webrtc {
namespace {

int TaskQueuePriorityToGCD(TaskQueueFactory::Priority priority) {
  switch (priority) {
    case TaskQueueFactory::Priority::NORMAL:
      return DISPATCH_QUEUE_PRIORITY_DEFAULT;
    case TaskQueueFactory::Priority::HIGH:
      return DISPATCH_QUEUE_PRIORITY_HIGH;
    case TaskQueueFactory::Priority::LOW:
      return DISPATCH_QUEUE_PRIORITY_LOW;
  }
}

class TaskQueueGcd final : public TaskQueueBase {
 public:
  TaskQueueGcd(absl::string_view queue_name, int gcd_priority);

  void Delete() override;

 protected:
  void PostTaskImpl(absl::AnyInvocable<void() &&> task,
                    const PostTaskTraits& traits,
                    const Location& location) override;
  void PostDelayedTaskImpl(absl::AnyInvocable<void() &&> task,
                           TimeDelta delay,
                           const PostDelayedTaskTraits& traits,
                           const Location& location) override;

 private:
  struct TaskContext {
    TaskContext(TaskQueueGcd* queue, absl::AnyInvocable<void() &&> task)
        : queue(queue), task(std::move(task)) {}

    TaskQueueGcd* const queue;
    absl::AnyInvocable<void() &&> task;
  };

  ~TaskQueueGcd() override;
  static void RunTask(void* task_context);
  static void SetNotActive(void* task_queue);
  static void DeleteQueue(void* task_queue);

  dispatch_queue_t queue_;
  bool is_active_;
};

TaskQueueGcd::TaskQueueGcd(absl::string_view queue_name, int gcd_priority)
    : queue_(RTCDispatchQueueCreateWithTarget(
          std::string(queue_name).c_str(),
          DISPATCH_QUEUE_SERIAL,
          dispatch_get_global_queue(gcd_priority, 0))),
      is_active_(true) {
  RTC_CHECK(queue_);
  dispatch_set_context(queue_, this);
  // Assign a finalizer that will delete the queue when the last reference
  // is released. This may run after the TaskQueue::Delete.
  dispatch_set_finalizer_f(queue_, &DeleteQueue);
}

TaskQueueGcd::~TaskQueueGcd() = default;

void TaskQueueGcd::Delete() {
  RTC_DCHECK(!IsCurrent());
  // Implementation/behavioral note:
  // Dispatch queues are reference counted via calls to dispatch_retain and
  // dispatch_release. Pending blocks submitted to a queue also hold a
  // reference to the queue until they have finished. Once all references to a
  // queue have been released, the queue will be deallocated by the system.
  // This is why we check the is_active_ before running tasks.

  // Use dispatch_sync to set the is_active_ to guarantee that there's not a
  // race with checking it from a task.
  dispatch_sync_f(queue_, this, &SetNotActive);
  dispatch_release(queue_);
}

void TaskQueueGcd::PostTaskImpl(absl::AnyInvocable<void() &&> task,
                                const PostTaskTraits& traits,
                                const Location& location) {
  auto* context = new TaskContext(this, std::move(task));
  dispatch_async_f(queue_, context, &RunTask);
}

void TaskQueueGcd::PostDelayedTaskImpl(absl::AnyInvocable<void() &&> task,
                                       TimeDelta delay,
                                       const PostDelayedTaskTraits& traits,
                                       const Location& location) {
  auto* context = new TaskContext(this, std::move(task));
  dispatch_after_f(dispatch_time(DISPATCH_TIME_NOW, delay.us() * NSEC_PER_USEC),
                   queue_, context, &RunTask);
}

// static
void TaskQueueGcd::RunTask(void* task_context) {
  std::unique_ptr<TaskContext> tc(static_cast<TaskContext*>(task_context));
  CurrentTaskQueueSetter set_current(tc->queue);
  if (tc->queue->is_active_) {
    std::move(tc->task)();
  }
  // Delete the task before CurrentTaskQueueSetter clears state that this code
  // is running on the task queue.
  tc = nullptr;
}

// static
void TaskQueueGcd::SetNotActive(void* task_queue) {
  static_cast<TaskQueueGcd*>(task_queue)->is_active_ = false;
}

// static
void TaskQueueGcd::DeleteQueue(void* task_queue) {
  delete static_cast<TaskQueueGcd*>(task_queue);
}

class TaskQueueGcdFactory final : public TaskQueueFactory {
 public:
  std::unique_ptr<TaskQueueBase, TaskQueueDeleter> CreateTaskQueue(
      absl::string_view name,
      Priority priority) const override {
    return std::unique_ptr<TaskQueueBase, TaskQueueDeleter>(
        new TaskQueueGcd(name, TaskQueuePriorityToGCD(priority)));
  }
};

}  // namespace

std::unique_ptr<TaskQueueFactory> CreateTaskQueueGcdFactory() {
  return std::make_unique<TaskQueueGcdFactory>();
}

}  // namespace webrtc
