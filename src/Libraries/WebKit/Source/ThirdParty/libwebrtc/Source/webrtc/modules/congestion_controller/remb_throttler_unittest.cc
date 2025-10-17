/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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
#include "modules/congestion_controller/remb_throttler.h"

#include <vector>

#include "api/units/data_rate.h"
#include "api/units/time_delta.h"
#include "system_wrappers/include/clock.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

using ::testing::_;
using ::testing::MockFunction;

TEST(RembThrottlerTest, CallRembSenderOnFirstReceiveBitrateChange) {
  SimulatedClock clock(Timestamp::Zero());
  MockFunction<void(uint64_t, std::vector<uint32_t>)> remb_sender;
  RembThrottler remb_throttler(remb_sender.AsStdFunction(), &clock);

  EXPECT_CALL(remb_sender, Call(12345, std::vector<uint32_t>({1, 2, 3})));
  remb_throttler.OnReceiveBitrateChanged({1, 2, 3}, /*bitrate_bps=*/12345);
}

TEST(RembThrottlerTest, ThrottlesSmallReceiveBitrateDecrease) {
  SimulatedClock clock(Timestamp::Zero());
  MockFunction<void(uint64_t, std::vector<uint32_t>)> remb_sender;
  RembThrottler remb_throttler(remb_sender.AsStdFunction(), &clock);

  EXPECT_CALL(remb_sender, Call);
  remb_throttler.OnReceiveBitrateChanged({1, 2, 3}, /*bitrate_bps=*/12346);
  clock.AdvanceTime(TimeDelta::Millis(100));
  remb_throttler.OnReceiveBitrateChanged({1, 2, 3}, /*bitrate_bps=*/12345);

  EXPECT_CALL(remb_sender, Call(12345, _));
  clock.AdvanceTime(TimeDelta::Millis(101));
  remb_throttler.OnReceiveBitrateChanged({1, 2, 3}, /*bitrate_bps=*/12345);
}

TEST(RembThrottlerTest, DoNotThrottleLargeReceiveBitrateDecrease) {
  SimulatedClock clock(Timestamp::Zero());
  MockFunction<void(uint64_t, std::vector<uint32_t>)> remb_sender;
  RembThrottler remb_throttler(remb_sender.AsStdFunction(), &clock);

  EXPECT_CALL(remb_sender, Call(2345, _));
  EXPECT_CALL(remb_sender, Call(1234, _));
  remb_throttler.OnReceiveBitrateChanged({1, 2, 3}, /*bitrate_bps=*/2345);
  clock.AdvanceTime(TimeDelta::Millis(1));
  remb_throttler.OnReceiveBitrateChanged({1, 2, 3}, /*bitrate_bps=*/1234);
}

TEST(RembThrottlerTest, ThrottlesReceiveBitrateIncrease) {
  SimulatedClock clock(Timestamp::Zero());
  MockFunction<void(uint64_t, std::vector<uint32_t>)> remb_sender;
  RembThrottler remb_throttler(remb_sender.AsStdFunction(), &clock);

  EXPECT_CALL(remb_sender, Call);
  remb_throttler.OnReceiveBitrateChanged({1, 2, 3}, /*bitrate_bps=*/1234);
  clock.AdvanceTime(TimeDelta::Millis(100));
  remb_throttler.OnReceiveBitrateChanged({1, 2, 3}, /*bitrate_bps=*/2345);

  // Updates 200ms after previous callback is not throttled.
  EXPECT_CALL(remb_sender, Call(2345, _));
  clock.AdvanceTime(TimeDelta::Millis(101));
  remb_throttler.OnReceiveBitrateChanged({1, 2, 3}, /*bitrate_bps=*/2345);
}

TEST(RembThrottlerTest, CallRembSenderOnSetMaxDesiredReceiveBitrate) {
  SimulatedClock clock(Timestamp::Zero());
  MockFunction<void(uint64_t, std::vector<uint32_t>)> remb_sender;
  RembThrottler remb_throttler(remb_sender.AsStdFunction(), &clock);
  EXPECT_CALL(remb_sender, Call(1234, _));
  remb_throttler.SetMaxDesiredReceiveBitrate(DataRate::BitsPerSec(1234));
}

TEST(RembThrottlerTest, CallRembSenderWithMinOfMaxDesiredAndOnReceivedBitrate) {
  SimulatedClock clock(Timestamp::Zero());
  MockFunction<void(uint64_t, std::vector<uint32_t>)> remb_sender;
  RembThrottler remb_throttler(remb_sender.AsStdFunction(), &clock);

  EXPECT_CALL(remb_sender, Call(1234, _));
  remb_throttler.OnReceiveBitrateChanged({1, 2, 3}, /*bitrate_bps=*/1234);
  clock.AdvanceTime(TimeDelta::Millis(1));
  remb_throttler.SetMaxDesiredReceiveBitrate(DataRate::BitsPerSec(4567));

  clock.AdvanceTime(TimeDelta::Millis(200));
  EXPECT_CALL(remb_sender, Call(4567, _));
  remb_throttler.OnReceiveBitrateChanged({1, 2, 3}, /*bitrate_bps=*/5678);
}

}  // namespace webrtc
