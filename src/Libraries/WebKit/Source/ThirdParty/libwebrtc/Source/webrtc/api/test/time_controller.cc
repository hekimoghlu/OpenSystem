/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
#include "api/test/time_controller.h"

#include <functional>
#include <memory>

#include "absl/strings/string_view.h"
#include "api/task_queue/task_queue_base.h"
#include "api/task_queue/task_queue_factory.h"
#include "api/units/time_delta.h"

namespace webrtc {
std::unique_ptr<TaskQueueFactory> TimeController::CreateTaskQueueFactory() {
  class FactoryWrapper final : public TaskQueueFactory {
   public:
    explicit FactoryWrapper(TaskQueueFactory* inner_factory)
        : inner_(inner_factory) {}
    std::unique_ptr<TaskQueueBase, TaskQueueDeleter> CreateTaskQueue(
        absl::string_view name,
        Priority priority) const override {
      return inner_->CreateTaskQueue(name, priority);
    }

   private:
    TaskQueueFactory* const inner_;
  };
  return std::make_unique<FactoryWrapper>(GetTaskQueueFactory());
}
bool TimeController::Wait(const std::function<bool()>& condition,
                          TimeDelta max_duration) {
  // Step size is chosen to be short enough to not significantly affect latency
  // in real time tests while being long enough to avoid adding too much load to
  // the system.
  const auto kStep = TimeDelta::Millis(5);
  for (auto elapsed = TimeDelta::Zero(); elapsed < max_duration;
       elapsed += kStep) {
    if (condition())
      return true;
    AdvanceTime(kStep);
  }
  return condition();
}
}  // namespace webrtc
