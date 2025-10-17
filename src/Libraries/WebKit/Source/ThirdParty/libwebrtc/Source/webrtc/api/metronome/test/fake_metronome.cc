/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
#include "api/metronome/test/fake_metronome.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "api/task_queue/task_queue_base.h"
#include "api/units/time_delta.h"

namespace webrtc::test {

ForcedTickMetronome::ForcedTickMetronome(TimeDelta tick_period)
    : tick_period_(tick_period) {}

void ForcedTickMetronome::RequestCallOnNextTick(
    absl::AnyInvocable<void() &&> callback) {
  callbacks_.push_back(std::move(callback));
}

TimeDelta ForcedTickMetronome::TickPeriod() const {
  return tick_period_;
}

size_t ForcedTickMetronome::NumListeners() {
  return callbacks_.size();
}

void ForcedTickMetronome::Tick() {
  std::vector<absl::AnyInvocable<void() &&>> callbacks;
  callbacks_.swap(callbacks);
  for (auto& callback : callbacks)
    std::move(callback)();
}

FakeMetronome::FakeMetronome(TimeDelta tick_period)
    : tick_period_(tick_period) {}

void FakeMetronome::SetTickPeriod(TimeDelta tick_period) {
  tick_period_ = tick_period;
}

void FakeMetronome::RequestCallOnNextTick(
    absl::AnyInvocable<void() &&> callback) {
  TaskQueueBase* current = TaskQueueBase::Current();
  callbacks_.push_back(std::move(callback));
  if (callbacks_.size() == 1) {
    current->PostDelayedTask(
        [this] {
          std::vector<absl::AnyInvocable<void() &&>> callbacks;
          callbacks_.swap(callbacks);
          for (auto& callback : callbacks)
            std::move(callback)();
        },
        tick_period_);
  }
}

TimeDelta FakeMetronome::TickPeriod() const {
  return tick_period_;
}

}  // namespace webrtc::test
