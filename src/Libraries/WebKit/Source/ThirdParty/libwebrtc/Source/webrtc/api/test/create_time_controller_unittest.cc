/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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
#include "api/test/create_time_controller.h"

#include <functional>

#include "api/test/time_controller.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "system_wrappers/include/clock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

class FakeAlarm : public ControlledAlarmClock {
 public:
  explicit FakeAlarm(Timestamp start_time);

  Clock* GetClock() override;
  bool ScheduleAlarmAt(Timestamp deadline) override;
  void SetCallback(std::function<void()> callback) override;
  void Sleep(TimeDelta duration) override;

 private:
  SimulatedClock clock_;
  Timestamp deadline_;
  std::function<void()> callback_;
};

FakeAlarm::FakeAlarm(Timestamp start_time)
    : clock_(start_time),
      deadline_(Timestamp::PlusInfinity()),
      callback_([] {}) {}

Clock* FakeAlarm::GetClock() {
  return &clock_;
}

bool FakeAlarm::ScheduleAlarmAt(Timestamp deadline) {
  if (deadline < deadline_) {
    deadline_ = deadline;
    return true;
  }
  return false;
}

void FakeAlarm::SetCallback(std::function<void()> callback) {
  callback_ = callback;
}

void FakeAlarm::Sleep(TimeDelta duration) {
  Timestamp end_time = clock_.CurrentTime() + duration;

  while (deadline_ <= end_time) {
    clock_.AdvanceTime(deadline_ - clock_.CurrentTime());
    deadline_ = Timestamp::PlusInfinity();
    callback_();
  }

  clock_.AdvanceTime(end_time - clock_.CurrentTime());
}

TEST(CreateTimeControllerTest, CreatesNonNullController) {
  FakeAlarm alarm(Timestamp::Millis(100));
  EXPECT_NE(CreateTimeController(&alarm), nullptr);
}

}  // namespace
}  // namespace webrtc
