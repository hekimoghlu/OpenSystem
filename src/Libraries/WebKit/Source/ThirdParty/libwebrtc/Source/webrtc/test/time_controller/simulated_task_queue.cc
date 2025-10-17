/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#include "test/time_controller/simulated_task_queue.h"

#include <algorithm>
#include <utility>

namespace webrtc {

SimulatedTaskQueue::SimulatedTaskQueue(
    sim_time_impl::SimulatedTimeControllerImpl* handler,
    absl::string_view name)
    : handler_(handler), name_(new char[name.size()]) {
  std::copy_n(name.begin(), name.size(), name_);
}

SimulatedTaskQueue::~SimulatedTaskQueue() {
  handler_->Unregister(this);
  delete[] name_;
}

void SimulatedTaskQueue::Delete() {
  // Need to destroy the tasks outside of the lock because task destruction
  // can lead to re-entry in SimulatedTaskQueue via custom destructors.
  std::deque<absl::AnyInvocable<void() &&>> ready_tasks;
  std::map<Timestamp, std::vector<absl::AnyInvocable<void() &&>>> delayed_tasks;
  {
    MutexLock lock(&lock_);
    ready_tasks_.swap(ready_tasks);
    delayed_tasks_.swap(delayed_tasks);
  }
  ready_tasks.clear();
  delayed_tasks.clear();
  delete this;
}

void SimulatedTaskQueue::RunReady(Timestamp at_time) {
  MutexLock lock(&lock_);
  for (auto it = delayed_tasks_.begin();
       it != delayed_tasks_.end() && it->first <= at_time;
       it = delayed_tasks_.erase(it)) {
    for (auto& task : it->second) {
      ready_tasks_.push_back(std::move(task));
    }
  }
  CurrentTaskQueueSetter set_current(this);
  while (!ready_tasks_.empty()) {
    absl::AnyInvocable<void()&&> ready = std::move(ready_tasks_.front());
    ready_tasks_.pop_front();
    lock_.Unlock();
    std::move(ready)();
    ready = nullptr;
    lock_.Lock();
  }
  if (!delayed_tasks_.empty()) {
    next_run_time_ = delayed_tasks_.begin()->first;
  } else {
    next_run_time_ = Timestamp::PlusInfinity();
  }
}

void SimulatedTaskQueue::PostTaskImpl(absl::AnyInvocable<void() &&> task,
                                      const PostTaskTraits& /*traits*/,
                                      const Location& /*location*/) {
  MutexLock lock(&lock_);
  ready_tasks_.push_back(std::move(task));
  next_run_time_ = Timestamp::MinusInfinity();
}

void SimulatedTaskQueue::PostDelayedTaskImpl(
    absl::AnyInvocable<void() &&> task,
    TimeDelta delay,
    const PostDelayedTaskTraits& /*traits*/,
    const Location& /*location*/) {
  MutexLock lock(&lock_);
  Timestamp target_time = handler_->CurrentTime() + delay;
  delayed_tasks_[target_time].push_back(std::move(task));
  next_run_time_ = std::min(next_run_time_, target_time);
}

}  // namespace webrtc
