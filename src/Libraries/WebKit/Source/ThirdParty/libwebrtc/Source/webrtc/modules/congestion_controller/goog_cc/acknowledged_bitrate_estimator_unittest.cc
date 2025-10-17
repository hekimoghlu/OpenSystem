/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
#include "modules/congestion_controller/goog_cc/acknowledged_bitrate_estimator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "api/transport/field_trial_based_config.h"
#include "api/transport/network_types.h"
#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/timestamp.h"
#include "modules/congestion_controller/goog_cc/bitrate_estimator.h"
#include "test/gmock.h"
#include "test/gtest.h"

using ::testing::InSequence;
using ::testing::Return;

namespace webrtc {

namespace {

constexpr int64_t kFirstArrivalTimeMs = 10;
constexpr int64_t kFirstSendTimeMs = 10;
constexpr uint16_t kSequenceNumber = 1;
constexpr size_t kPayloadSize = 10;

class MockBitrateEstimator : public BitrateEstimator {
 public:
  using BitrateEstimator::BitrateEstimator;
  MOCK_METHOD(void,
              Update,
              (Timestamp at_time, DataSize data_size, bool in_alr),
              (override));
  MOCK_METHOD(std::optional<DataRate>, bitrate, (), (const, override));
  MOCK_METHOD(void, ExpectFastRateChange, (), (override));
};

struct AcknowledgedBitrateEstimatorTestStates {
  FieldTrialBasedConfig field_trial_config;
  std::unique_ptr<AcknowledgedBitrateEstimator> acknowledged_bitrate_estimator;
  MockBitrateEstimator* mock_bitrate_estimator;
};

AcknowledgedBitrateEstimatorTestStates CreateTestStates() {
  AcknowledgedBitrateEstimatorTestStates states;
  auto mock_bitrate_estimator =
      std::make_unique<MockBitrateEstimator>(&states.field_trial_config);
  states.mock_bitrate_estimator = mock_bitrate_estimator.get();
  states.acknowledged_bitrate_estimator =
      std::make_unique<AcknowledgedBitrateEstimator>(
          &states.field_trial_config, std::move(mock_bitrate_estimator));
  return states;
}

std::vector<PacketResult> CreateFeedbackVector() {
  std::vector<PacketResult> packet_feedback_vector(2);
  packet_feedback_vector[0].receive_time =
      Timestamp::Millis(kFirstArrivalTimeMs);
  packet_feedback_vector[0].sent_packet.send_time =
      Timestamp::Millis(kFirstSendTimeMs);
  packet_feedback_vector[0].sent_packet.sequence_number = kSequenceNumber;
  packet_feedback_vector[0].sent_packet.size = DataSize::Bytes(kPayloadSize);
  packet_feedback_vector[1].receive_time =
      Timestamp::Millis(kFirstArrivalTimeMs + 10);
  packet_feedback_vector[1].sent_packet.send_time =
      Timestamp::Millis(kFirstSendTimeMs + 10);
  packet_feedback_vector[1].sent_packet.sequence_number = kSequenceNumber;
  packet_feedback_vector[1].sent_packet.size =
      DataSize::Bytes(kPayloadSize + 10);
  return packet_feedback_vector;
}

}  // anonymous namespace

TEST(TestAcknowledgedBitrateEstimator, UpdateBandwidth) {
  auto states = CreateTestStates();
  auto packet_feedback_vector = CreateFeedbackVector();
  {
    InSequence dummy;
    EXPECT_CALL(*states.mock_bitrate_estimator,
                Update(packet_feedback_vector[0].receive_time,
                       packet_feedback_vector[0].sent_packet.size,
                       /*in_alr*/ false))
        .Times(1);
    EXPECT_CALL(*states.mock_bitrate_estimator,
                Update(packet_feedback_vector[1].receive_time,
                       packet_feedback_vector[1].sent_packet.size,
                       /*in_alr*/ false))
        .Times(1);
  }
  states.acknowledged_bitrate_estimator->IncomingPacketFeedbackVector(
      packet_feedback_vector);
}

TEST(TestAcknowledgedBitrateEstimator, ExpectFastRateChangeWhenLeftAlr) {
  auto states = CreateTestStates();
  auto packet_feedback_vector = CreateFeedbackVector();
  {
    InSequence dummy;
    EXPECT_CALL(*states.mock_bitrate_estimator,
                Update(packet_feedback_vector[0].receive_time,
                       packet_feedback_vector[0].sent_packet.size,
                       /*in_alr*/ false))
        .Times(1);
    EXPECT_CALL(*states.mock_bitrate_estimator, ExpectFastRateChange())
        .Times(1);
    EXPECT_CALL(*states.mock_bitrate_estimator,
                Update(packet_feedback_vector[1].receive_time,
                       packet_feedback_vector[1].sent_packet.size,
                       /*in_alr*/ false))
        .Times(1);
  }
  states.acknowledged_bitrate_estimator->SetAlrEndedTime(
      Timestamp::Millis(kFirstArrivalTimeMs + 1));
  states.acknowledged_bitrate_estimator->IncomingPacketFeedbackVector(
      packet_feedback_vector);
}

TEST(TestAcknowledgedBitrateEstimator, ReturnBitrate) {
  auto states = CreateTestStates();
  std::optional<DataRate> return_value = DataRate::KilobitsPerSec(42);
  EXPECT_CALL(*states.mock_bitrate_estimator, bitrate())
      .Times(1)
      .WillOnce(Return(return_value));
  EXPECT_EQ(return_value, states.acknowledged_bitrate_estimator->bitrate());
}

}  // namespace webrtc*/
