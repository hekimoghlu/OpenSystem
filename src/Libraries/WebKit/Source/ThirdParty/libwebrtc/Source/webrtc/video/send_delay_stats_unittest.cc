/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
#include "video/send_delay_stats.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "call/rtp_config.h"
#include "system_wrappers/include/metrics.h"
#include "test/gtest.h"

namespace webrtc {
namespace {
const uint32_t kSsrc1 = 17;
const uint32_t kSsrc2 = 42;
const uint32_t kRtxSsrc1 = 18;
const uint32_t kRtxSsrc2 = 43;
const uint16_t kPacketId = 2345;
const TimeDelta kMaxPacketDelay = TimeDelta::Seconds(11);
const int kMinRequiredPeriodicSamples = 5;
const int kProcessIntervalMs = 2000;
}  // namespace

class SendDelayStatsTest : public ::testing::Test {
 public:
  SendDelayStatsTest() : clock_(1234), config_(CreateConfig()) {}
  virtual ~SendDelayStatsTest() {}

 protected:
  virtual void SetUp() {
    stats_.reset(new SendDelayStats(&clock_));
    stats_->AddSsrcs(config_);
  }

  VideoSendStream::Config CreateConfig() {
    VideoSendStream::Config config(nullptr);
    config.rtp.ssrcs.push_back(kSsrc1);
    config.rtp.ssrcs.push_back(kSsrc2);
    config.rtp.rtx.ssrcs.push_back(kRtxSsrc1);
    config.rtp.rtx.ssrcs.push_back(kRtxSsrc2);
    return config;
  }

  void OnSendPacket(uint16_t id, uint32_t ssrc) {
    OnSendPacket(id, ssrc, clock_.CurrentTime());
  }

  void OnSendPacket(uint16_t id, uint32_t ssrc, Timestamp capture) {
    stats_->OnSendPacket(id, capture, ssrc);
  }

  bool OnSentPacket(uint16_t id) {
    return stats_->OnSentPacket(id, clock_.CurrentTime());
  }

  SimulatedClock clock_;
  VideoSendStream::Config config_;
  std::unique_ptr<SendDelayStats> stats_;
};

TEST_F(SendDelayStatsTest, SentPacketFound) {
  EXPECT_FALSE(OnSentPacket(kPacketId));
  OnSendPacket(kPacketId, kSsrc1);
  EXPECT_TRUE(OnSentPacket(kPacketId));   // Packet found.
  EXPECT_FALSE(OnSentPacket(kPacketId));  // Packet removed when found.
}

TEST_F(SendDelayStatsTest, SentPacketNotFoundForNonRegisteredSsrc) {
  OnSendPacket(kPacketId, kSsrc1);
  EXPECT_TRUE(OnSentPacket(kPacketId));
  OnSendPacket(kPacketId + 1, kSsrc2);
  EXPECT_TRUE(OnSentPacket(kPacketId + 1));
  OnSendPacket(kPacketId + 2, kRtxSsrc1);  // RTX SSRC not registered.
  EXPECT_FALSE(OnSentPacket(kPacketId + 2));
}

TEST_F(SendDelayStatsTest, SentPacketFoundWithMaxSendDelay) {
  OnSendPacket(kPacketId, kSsrc1);
  clock_.AdvanceTime(kMaxPacketDelay - TimeDelta::Millis(1));
  OnSendPacket(kPacketId + 1, kSsrc1);       // kPacketId -> not old/removed.
  EXPECT_TRUE(OnSentPacket(kPacketId));      // Packet found.
  EXPECT_TRUE(OnSentPacket(kPacketId + 1));  // Packet found.
}

TEST_F(SendDelayStatsTest, OldPacketsRemoved) {
  const Timestamp kCaptureTime = clock_.CurrentTime();
  OnSendPacket(0xffffu, kSsrc1, kCaptureTime);
  OnSendPacket(0u, kSsrc1, kCaptureTime);
  OnSendPacket(1u, kSsrc1, kCaptureTime + TimeDelta::Millis(1));
  clock_.AdvanceTime(kMaxPacketDelay);  // 0xffff, 0 -> old.
  OnSendPacket(2u, kSsrc1, kCaptureTime + TimeDelta::Millis(2));

  EXPECT_FALSE(OnSentPacket(0xffffu));  // Old removed.
  EXPECT_FALSE(OnSentPacket(0u));       // Old removed.
  EXPECT_TRUE(OnSentPacket(1u));
  EXPECT_TRUE(OnSentPacket(2u));
}

TEST_F(SendDelayStatsTest, HistogramsAreUpdated) {
  metrics::Reset();
  const int64_t kDelayMs1 = 5;
  const int64_t kDelayMs2 = 15;
  const int kNumSamples = kMinRequiredPeriodicSamples * kProcessIntervalMs /
                              (kDelayMs1 + kDelayMs2) +
                          1;

  uint16_t id = 0;
  for (int i = 0; i < kNumSamples; ++i) {
    OnSendPacket(++id, kSsrc1);
    clock_.AdvanceTimeMilliseconds(kDelayMs1);
    EXPECT_TRUE(OnSentPacket(id));
    OnSendPacket(++id, kSsrc2);
    clock_.AdvanceTimeMilliseconds(kDelayMs2);
    EXPECT_TRUE(OnSentPacket(id));
  }
  stats_.reset();
  EXPECT_METRIC_EQ(2, metrics::NumSamples("WebRTC.Video.SendDelayInMs"));
  EXPECT_METRIC_EQ(1,
                   metrics::NumEvents("WebRTC.Video.SendDelayInMs", kDelayMs1));
  EXPECT_METRIC_EQ(1,
                   metrics::NumEvents("WebRTC.Video.SendDelayInMs", kDelayMs2));
}

}  // namespace webrtc
