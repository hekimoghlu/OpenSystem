/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#include "rtc_base/gunit.h"
#include "rtc_base/task_queue_for_test.h"
#include "system_wrappers/include/sleep.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {

using NonSenderRttTest = CallTest;

TEST_F(NonSenderRttTest, NonSenderRttStats) {
  class NonSenderRttTest : public AudioEndToEndTest {
   public:
    const int kLongTimeoutMs = 20000;
    const int64_t kRttMs = 30;

    explicit NonSenderRttTest(TaskQueueBase* task_queue)
        : task_queue_(task_queue) {}

    BuiltInNetworkBehaviorConfig GetSendTransportConfig() const override {
      BuiltInNetworkBehaviorConfig pipe_config;
      pipe_config.queue_delay_ms = kRttMs / 2;
      return pipe_config;
    }

    void ModifyAudioConfigs(AudioSendStream::Config* send_config,
                            std::vector<AudioReceiveStreamInterface::Config>*
                                receive_configs) override {
      ASSERT_EQ(receive_configs->size(), 1U);
      (*receive_configs)[0].enable_non_sender_rtt = true;
      AudioEndToEndTest::ModifyAudioConfigs(send_config, receive_configs);
      send_config->send_codec_spec->enable_non_sender_rtt = true;
    }

    void PerformTest() override {
      // Wait until we have an RTT measurement, but no longer than
      // `kLongTimeoutMs`. This usually takes around 5 seconds, but in rare
      // cases it can take more than 10 seconds.
      EXPECT_TRUE_WAIT(HasRoundTripTimeMeasurement(), kLongTimeoutMs);
    }

    void OnStreamsStopped() override {
      AudioReceiveStreamInterface::Stats recv_stats =
          receive_stream()->GetStats(/*get_and_clear_legacy_stats=*/true);
      EXPECT_GT(recv_stats.round_trip_time_measurements, 0);
      ASSERT_TRUE(recv_stats.round_trip_time.has_value());
      EXPECT_GT(recv_stats.round_trip_time->ms(), 0);
      EXPECT_GE(recv_stats.total_round_trip_time.ms(),
                recv_stats.round_trip_time->ms());
    }

   protected:
    bool HasRoundTripTimeMeasurement() {
      bool has_rtt = false;
      // GetStats() can only be called on `task_queue_`, block while we check.
      SendTask(task_queue_, [this, &has_rtt]() {
        if (receive_stream() &&
            receive_stream()->GetStats(true).round_trip_time_measurements > 0) {
          has_rtt = true;
        }
      });
      return has_rtt;
    }

   private:
    TaskQueueBase* task_queue_;
  } test(task_queue());

  RunBaseTest(&test);
}

}  // namespace test
}  // namespace webrtc
