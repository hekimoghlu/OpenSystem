/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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
#include "net/dcsctp/timer/task_queue_timeout.h"

#include <memory>

#include "api/task_queue/task_queue_base.h"
#include "api/task_queue/test/mock_task_queue_base.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"
#include "test/time_controller/simulated_time_controller.h"

namespace dcsctp {
namespace {
using ::testing::_;
using ::testing::Field;
using ::testing::MockFunction;
using ::testing::NiceMock;

class TaskQueueTimeoutTest : public testing::Test {
 protected:
  TaskQueueTimeoutTest()
      : time_controller_(webrtc::Timestamp::Millis(1234)),
        task_queue_(time_controller_.GetMainThread()),
        factory_(
            *task_queue_,
            [this]() {
              return TimeMs(time_controller_.GetClock()->CurrentTime().ms());
            },
            on_expired_.AsStdFunction()) {}

  void AdvanceTime(DurationMs duration) {
    time_controller_.AdvanceTime(webrtc::TimeDelta::Millis(*duration));
  }

  MockFunction<void(TimeoutID)> on_expired_;
  webrtc::GlobalSimulatedTimeController time_controller_;

  rtc::Thread* task_queue_;
  TaskQueueTimeoutFactory factory_;
};

TEST_F(TaskQueueTimeoutTest, StartPostsDelayedTask) {
  std::unique_ptr<Timeout> timeout = factory_.CreateTimeout();
  timeout->Start(DurationMs(1000), TimeoutID(1));

  EXPECT_CALL(on_expired_, Call).Times(0);
  AdvanceTime(DurationMs(999));

  EXPECT_CALL(on_expired_, Call(TimeoutID(1)));
  AdvanceTime(DurationMs(1));
}

TEST_F(TaskQueueTimeoutTest, StopBeforeExpiringDoesntTrigger) {
  std::unique_ptr<Timeout> timeout = factory_.CreateTimeout();
  timeout->Start(DurationMs(1000), TimeoutID(1));

  EXPECT_CALL(on_expired_, Call).Times(0);
  AdvanceTime(DurationMs(999));

  timeout->Stop();

  AdvanceTime(DurationMs(1));
  AdvanceTime(DurationMs(1000));
}

TEST_F(TaskQueueTimeoutTest, RestartPrologingTimeoutDuration) {
  std::unique_ptr<Timeout> timeout = factory_.CreateTimeout();
  timeout->Start(DurationMs(1000), TimeoutID(1));

  EXPECT_CALL(on_expired_, Call).Times(0);
  AdvanceTime(DurationMs(500));

  timeout->Restart(DurationMs(1000), TimeoutID(2));

  AdvanceTime(DurationMs(999));

  EXPECT_CALL(on_expired_, Call(TimeoutID(2)));
  AdvanceTime(DurationMs(1));
}

TEST_F(TaskQueueTimeoutTest, RestartWithShorterDurationExpiresWhenExpected) {
  std::unique_ptr<Timeout> timeout = factory_.CreateTimeout();
  timeout->Start(DurationMs(1000), TimeoutID(1));

  EXPECT_CALL(on_expired_, Call).Times(0);
  AdvanceTime(DurationMs(500));

  timeout->Restart(DurationMs(200), TimeoutID(2));

  AdvanceTime(DurationMs(199));

  EXPECT_CALL(on_expired_, Call(TimeoutID(2)));
  AdvanceTime(DurationMs(1));

  EXPECT_CALL(on_expired_, Call).Times(0);
  AdvanceTime(DurationMs(1000));
}

TEST_F(TaskQueueTimeoutTest, KilledBeforeExpired) {
  std::unique_ptr<Timeout> timeout = factory_.CreateTimeout();
  timeout->Start(DurationMs(1000), TimeoutID(1));

  EXPECT_CALL(on_expired_, Call).Times(0);
  AdvanceTime(DurationMs(500));

  timeout = nullptr;

  EXPECT_CALL(on_expired_, Call).Times(0);
  AdvanceTime(DurationMs(1000));
}

TEST(TaskQueueTimeoutWithMockTaskQueueTest, CanSetTimeoutPrecisionToLow) {
  NiceMock<webrtc::MockTaskQueueBase> mock_task_queue;
  EXPECT_CALL(
      mock_task_queue,
      PostDelayedTaskImpl(
          _, _,
          Field(
              &webrtc::MockTaskQueueBase::PostDelayedTaskTraits::high_precision,
              false),
          _));
  TaskQueueTimeoutFactory factory(
      mock_task_queue, []() { return TimeMs(1337); },
      [](TimeoutID timeout_id) {});
  std::unique_ptr<Timeout> timeout =
      factory.CreateTimeout(webrtc::TaskQueueBase::DelayPrecision::kLow);
  timeout->Start(DurationMs(1), TimeoutID(1));
}

TEST(TaskQueueTimeoutWithMockTaskQueueTest, CanSetTimeoutPrecisionToHigh) {
  NiceMock<webrtc::MockTaskQueueBase> mock_task_queue;
  EXPECT_CALL(
      mock_task_queue,
      PostDelayedTaskImpl(
          _, _,
          Field(
              &webrtc::MockTaskQueueBase::PostDelayedTaskTraits::high_precision,
              true),
          _));
  TaskQueueTimeoutFactory factory(
      mock_task_queue, []() { return TimeMs(1337); },
      [](TimeoutID timeout_id) {});
  std::unique_ptr<Timeout> timeout =
      factory.CreateTimeout(webrtc::TaskQueueBase::DelayPrecision::kHigh);
  timeout->Start(DurationMs(1), TimeoutID(1));
}

TEST(TaskQueueTimeoutWithMockTaskQueueTest, TimeoutPrecisionIsLowByDefault) {
  NiceMock<webrtc::MockTaskQueueBase> mock_task_queue;
  EXPECT_CALL(
      mock_task_queue,
      PostDelayedTaskImpl(
          _, _,
          Field(
              &webrtc::MockTaskQueueBase::PostDelayedTaskTraits::high_precision,
              false),
          _));
  TaskQueueTimeoutFactory factory(
      mock_task_queue, []() { return TimeMs(1337); },
      [](TimeoutID timeout_id) {});
  std::unique_ptr<Timeout> timeout = factory.CreateTimeout();
  timeout->Start(DurationMs(1), TimeoutID(1));
}

}  // namespace
}  // namespace dcsctp
