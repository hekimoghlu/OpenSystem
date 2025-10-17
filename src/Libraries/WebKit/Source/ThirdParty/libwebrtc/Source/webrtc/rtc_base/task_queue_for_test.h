/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
#ifndef RTC_BASE_TASK_QUEUE_FOR_TEST_H_
#define RTC_BASE_TASK_QUEUE_FOR_TEST_H_

#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/string_view.h"
#include "api/function_view.h"
#include "api/task_queue/task_queue_base.h"
#include "api/task_queue/task_queue_factory.h"
#include "rtc_base/checks.h"
#include "rtc_base/event.h"

namespace webrtc {

inline void SendTask(TaskQueueBase* task_queue,
                     rtc::FunctionView<void()> task) {
  if (task_queue->IsCurrent()) {
    task();
    return;
  }

  rtc::Event event;
  absl::Cleanup cleanup = [&event] { event.Set(); };
  task_queue->PostTask([task, cleanup = std::move(cleanup)] { task(); });
  RTC_CHECK(event.Wait(/*give_up_after=*/rtc::Event::kForever,
                       /*warn_after=*/TimeDelta::Seconds(10)));
}

class TaskQueueForTest {
 public:
  explicit TaskQueueForTest(
      std::unique_ptr<TaskQueueBase, TaskQueueDeleter> task_queue);
  explicit TaskQueueForTest(
      absl::string_view name = "TestQueue",
      TaskQueueFactory::Priority priority = TaskQueueFactory::Priority::NORMAL);
  TaskQueueForTest(const TaskQueueForTest&) = delete;
  TaskQueueForTest& operator=(const TaskQueueForTest&) = delete;
  ~TaskQueueForTest();

  bool IsCurrent() const { return impl_->IsCurrent(); }

  // Returns non-owning pointer to the task queue implementation.
  TaskQueueBase* Get() { return impl_.get(); }

  void PostTask(
      absl::AnyInvocable<void() &&> task,
      const webrtc::Location& location = webrtc::Location::Current()) {
    impl_->PostTask(std::move(task), location);
  }
  void PostDelayedTask(
      absl::AnyInvocable<void() &&> task,
      webrtc::TimeDelta delay,
      const webrtc::Location& location = webrtc::Location::Current()) {
    impl_->PostDelayedTask(std::move(task), delay, location);
  }
  void PostDelayedHighPrecisionTask(
      absl::AnyInvocable<void() &&> task,
      webrtc::TimeDelta delay,
      const webrtc::Location& location = webrtc::Location::Current()) {
    impl_->PostDelayedHighPrecisionTask(std::move(task), delay, location);
  }

  // A convenience, test-only method that blocks the current thread while
  // a task executes on the task queue.
  void SendTask(rtc::FunctionView<void()> task) {
    ::webrtc::SendTask(Get(), task);
  }

  // Wait for the completion of all tasks posted prior to the
  // WaitForPreviouslyPostedTasks() call.
  void WaitForPreviouslyPostedTasks() {
    RTC_DCHECK(!Get()->IsCurrent());
    // Post an empty task on the queue and wait for it to finish, to ensure
    // that all already posted tasks on the queue get executed.
    SendTask([]() {});
  }

 private:
  std::unique_ptr<TaskQueueBase, TaskQueueDeleter> impl_;
};

}  // namespace webrtc

#endif  // RTC_BASE_TASK_QUEUE_FOR_TEST_H_
