/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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
#include "video/task_queue_frame_decode_scheduler.h"

#include <stddef.h>

#include <memory>
#include <optional>
#include <utility>

#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "test/gmock.h"
#include "test/gtest.h"
#include "test/time_controller/simulated_time_controller.h"

namespace webrtc {

using ::testing::_;
using ::testing::Eq;
using ::testing::MockFunction;
using ::testing::Optional;

TEST(TaskQueueFrameDecodeSchedulerTest, FrameYieldedAfterSpecifiedPeriod) {
  GlobalSimulatedTimeController time_controller_(Timestamp::Seconds(2000));
  TaskQueueFrameDecodeScheduler scheduler(time_controller_.GetClock(),
                                          time_controller_.GetMainThread());
  constexpr TimeDelta decode_delay = TimeDelta::Millis(5);

  const Timestamp now = time_controller_.GetClock()->CurrentTime();
  const uint32_t rtp = 90000;
  const Timestamp render_time = now + TimeDelta::Millis(15);
  FrameDecodeTiming::FrameSchedule schedule = {
      .latest_decode_time = now + decode_delay, .render_time = render_time};

  MockFunction<void(uint32_t, Timestamp)> ready_cb;
  scheduler.ScheduleFrame(rtp, schedule, ready_cb.AsStdFunction());
  EXPECT_CALL(ready_cb, Call(_, _)).Times(0);
  EXPECT_THAT(scheduler.ScheduledRtpTimestamp(), Optional(rtp));
  time_controller_.AdvanceTime(TimeDelta::Zero());
  // Check that `ready_cb` has not been invoked yet.
  ::testing::Mock::VerifyAndClearExpectations(&ready_cb);

  EXPECT_CALL(ready_cb, Call(rtp, render_time)).Times(1);
  time_controller_.AdvanceTime(decode_delay);

  scheduler.Stop();
}

TEST(TaskQueueFrameDecodeSchedulerTest, NegativeDecodeDelayIsRoundedToZero) {
  GlobalSimulatedTimeController time_controller_(Timestamp::Seconds(2000));
  TaskQueueFrameDecodeScheduler scheduler(time_controller_.GetClock(),
                                          time_controller_.GetMainThread());
  constexpr TimeDelta decode_delay = TimeDelta::Millis(-5);
  const Timestamp now = time_controller_.GetClock()->CurrentTime();
  const uint32_t rtp = 90000;
  const Timestamp render_time = now + TimeDelta::Millis(15);
  FrameDecodeTiming::FrameSchedule schedule = {
      .latest_decode_time = now + decode_delay, .render_time = render_time};

  MockFunction<void(uint32_t, Timestamp)> ready_cb;
  EXPECT_CALL(ready_cb, Call(rtp, render_time)).Times(1);
  scheduler.ScheduleFrame(rtp, schedule, ready_cb.AsStdFunction());
  EXPECT_THAT(scheduler.ScheduledRtpTimestamp(), Optional(rtp));
  time_controller_.AdvanceTime(TimeDelta::Zero());

  scheduler.Stop();
}

TEST(TaskQueueFrameDecodeSchedulerTest, CancelOutstanding) {
  GlobalSimulatedTimeController time_controller_(Timestamp::Seconds(2000));
  TaskQueueFrameDecodeScheduler scheduler(time_controller_.GetClock(),
                                          time_controller_.GetMainThread());
  constexpr TimeDelta decode_delay = TimeDelta::Millis(50);
  const Timestamp now = time_controller_.GetClock()->CurrentTime();
  const uint32_t rtp = 90000;
  FrameDecodeTiming::FrameSchedule schedule = {
      .latest_decode_time = now + decode_delay,
      .render_time = now + TimeDelta::Millis(75)};

  MockFunction<void(uint32_t, Timestamp)> ready_cb;
  EXPECT_CALL(ready_cb, Call).Times(0);
  scheduler.ScheduleFrame(rtp, schedule, ready_cb.AsStdFunction());
  EXPECT_THAT(scheduler.ScheduledRtpTimestamp(), Optional(rtp));
  time_controller_.AdvanceTime(decode_delay / 2);
  EXPECT_THAT(scheduler.ScheduledRtpTimestamp(), Optional(rtp));
  scheduler.CancelOutstanding();
  EXPECT_THAT(scheduler.ScheduledRtpTimestamp(), Eq(std::nullopt));
  time_controller_.AdvanceTime(decode_delay / 2);

  scheduler.Stop();
}

}  // namespace webrtc
