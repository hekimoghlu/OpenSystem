/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#include "api/transport/test/create_feedback_generator.h"
#include "test/gtest.h"

namespace webrtc {
TEST(FeedbackGeneratorTest, ReportsFeedbackForSentPackets) {
  size_t kPacketSize = 1000;
  auto gen = CreateFeedbackGenerator(FeedbackGenerator::Config());
  for (int i = 0; i < 10; ++i) {
    gen->SendPacket(kPacketSize);
    gen->Sleep(TimeDelta::Millis(50));
  }
  auto feedback_list = gen->PopFeedback();
  EXPECT_GT(feedback_list.size(), 0u);
  for (const auto& feedback : feedback_list) {
    EXPECT_GT(feedback.packet_feedbacks.size(), 0u);
    for (const auto& packet : feedback.packet_feedbacks) {
      EXPECT_EQ(packet.sent_packet.size.bytes<size_t>(), kPacketSize);
    }
  }
}

TEST(FeedbackGeneratorTest, FeedbackIncludesLostPackets) {
  size_t kPacketSize = 1000;
  auto gen = CreateFeedbackGenerator(FeedbackGenerator::Config());
  BuiltInNetworkBehaviorConfig send_config_with_loss;
  send_config_with_loss.loss_percent = 50;
  gen->SetSendConfig(send_config_with_loss);
  for (int i = 0; i < 20; ++i) {
    gen->SendPacket(kPacketSize);
    gen->Sleep(TimeDelta::Millis(5));
  }
  auto feedback_list = gen->PopFeedback();
  ASSERT_GT(feedback_list.size(), 0u);
  EXPECT_NEAR(feedback_list[0].LostWithSendInfo().size(),
              feedback_list[0].ReceivedWithSendInfo().size(), 2);
}
}  // namespace webrtc
