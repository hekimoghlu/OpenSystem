/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 16, 2023.
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
#include "audio/test/audio_end_to_end_test.h"
#include "system_wrappers/include/sleep.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {

using NackTest = CallTest;

TEST_F(NackTest, ShouldNackInLossyNetwork) {
  class NackTest : public AudioEndToEndTest {
   public:
    const int kTestDurationMs = 2000;
    const int64_t kRttMs = 30;
    const int64_t kLossPercent = 30;
    const int kNackHistoryMs = 1000;

    BuiltInNetworkBehaviorConfig GetSendTransportConfig() const override {
      BuiltInNetworkBehaviorConfig pipe_config;
      pipe_config.queue_delay_ms = kRttMs / 2;
      pipe_config.loss_percent = kLossPercent;
      return pipe_config;
    }

    void ModifyAudioConfigs(AudioSendStream::Config* send_config,
                            std::vector<AudioReceiveStreamInterface::Config>*
                                receive_configs) override {
      ASSERT_EQ(receive_configs->size(), 1U);
      (*receive_configs)[0].rtp.nack.rtp_history_ms = kNackHistoryMs;
      AudioEndToEndTest::ModifyAudioConfigs(send_config, receive_configs);
    }

    void PerformTest() override { SleepMs(kTestDurationMs); }

    void OnStreamsStopped() override {
      AudioReceiveStreamInterface::Stats recv_stats =
          receive_stream()->GetStats(/*get_and_clear_legacy_stats=*/true);
      EXPECT_GT(recv_stats.nacks_sent, 0U);
      AudioSendStream::Stats send_stats = send_stream()->GetStats();
      EXPECT_GT(send_stats.retransmitted_packets_sent, 0U);
      EXPECT_GT(send_stats.nacks_received, 0U);
    }
  } test;

  RunBaseTest(&test);
}

}  // namespace test
}  // namespace webrtc
