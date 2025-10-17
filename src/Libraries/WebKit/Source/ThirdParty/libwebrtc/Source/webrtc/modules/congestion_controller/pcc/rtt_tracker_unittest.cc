/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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
#include "modules/congestion_controller/pcc/rtt_tracker.h"

#include "test/gtest.h"

namespace webrtc {
namespace pcc {
namespace test {
namespace {
const TimeDelta kInitialRtt = TimeDelta::Micros(10);
constexpr double kAlpha = 0.9;
const Timestamp kStartTime = Timestamp::Seconds(0);

PacketResult GetPacketWithRtt(TimeDelta rtt) {
  SentPacket packet;
  packet.send_time = kStartTime;
  PacketResult packet_result;
  packet_result.sent_packet = packet;
  if (rtt.IsFinite()) {
    packet_result.receive_time = kStartTime + rtt;
  } else {
    packet_result.receive_time = Timestamp::PlusInfinity();
  }
  return packet_result;
}
}  // namespace

TEST(PccRttTrackerTest, InitialValue) {
  RttTracker tracker{kInitialRtt, kAlpha};
  EXPECT_EQ(kInitialRtt, tracker.GetRtt());
  for (int i = 0; i < 100; ++i) {
    tracker.OnPacketsFeedback({GetPacketWithRtt(kInitialRtt)},
                              kStartTime + kInitialRtt);
  }
  EXPECT_EQ(kInitialRtt, tracker.GetRtt());
}

TEST(PccRttTrackerTest, DoNothingWhenPacketIsLost) {
  RttTracker tracker{kInitialRtt, kAlpha};
  tracker.OnPacketsFeedback({GetPacketWithRtt(TimeDelta::PlusInfinity())},
                            kStartTime + kInitialRtt);
  EXPECT_EQ(tracker.GetRtt(), kInitialRtt);
}

TEST(PccRttTrackerTest, ChangeInRtt) {
  RttTracker tracker{kInitialRtt, kAlpha};
  const TimeDelta kNewRtt = TimeDelta::Micros(100);
  tracker.OnPacketsFeedback({GetPacketWithRtt(kNewRtt)}, kStartTime + kNewRtt);
  EXPECT_GT(tracker.GetRtt(), kInitialRtt);
  EXPECT_LE(tracker.GetRtt(), kNewRtt);
  for (int i = 0; i < 100; ++i) {
    tracker.OnPacketsFeedback({GetPacketWithRtt(kNewRtt)},
                              kStartTime + kNewRtt);
  }
  const TimeDelta absolute_error = TimeDelta::Micros(1);
  EXPECT_NEAR(tracker.GetRtt().us(), kNewRtt.us(), absolute_error.us());
  EXPECT_LE(tracker.GetRtt(), kNewRtt);
}

}  // namespace test
}  // namespace pcc
}  // namespace webrtc
