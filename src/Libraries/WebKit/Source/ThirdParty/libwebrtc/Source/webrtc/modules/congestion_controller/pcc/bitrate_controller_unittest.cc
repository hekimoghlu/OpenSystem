/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#include "modules/congestion_controller/pcc/bitrate_controller.h"

#include <memory>
#include <utility>

#include "modules/congestion_controller/pcc/monitor_interval.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace pcc {
namespace test {
namespace {
constexpr double kInitialConversionFactor = 1;
constexpr double kInitialDynamicBoundary = 0.05;
constexpr double kDynamicBoundaryIncrement = 0.1;

constexpr double kDelayGradientCoefficient = 900;
constexpr double kLossCoefficient = 11.35;
constexpr double kThroughputCoefficient = 500 * 1000;
constexpr double kThroughputPower = 0.99;
constexpr double kDelayGradientThreshold = 0.01;
constexpr double kDelayGradientNegativeBound = 10;

const DataRate kTargetSendingRate = DataRate::KilobitsPerSec(300);
const double kEpsilon = 0.05;
const Timestamp kStartTime = Timestamp::Micros(0);
const TimeDelta kPacketsDelta = TimeDelta::Millis(1);
const TimeDelta kIntervalDuration = TimeDelta::Millis(1000);
const TimeDelta kDefaultRtt = TimeDelta::Millis(1000);
const DataSize kDefaultDataSize = DataSize::Bytes(100);

std::vector<PacketResult> CreatePacketResults(
    const std::vector<Timestamp>& packets_send_times,
    const std::vector<Timestamp>& packets_received_times = {},
    const std::vector<DataSize>& packets_sizes = {}) {
  std::vector<PacketResult> packet_results;
  PacketResult packet_result;
  SentPacket sent_packet;
  for (size_t i = 0; i < packets_send_times.size(); ++i) {
    sent_packet.send_time = packets_send_times[i];
    if (packets_sizes.empty()) {
      sent_packet.size = kDefaultDataSize;
    } else {
      sent_packet.size = packets_sizes[i];
    }
    packet_result.sent_packet = sent_packet;
    if (packets_received_times.empty()) {
      packet_result.receive_time = packets_send_times[i] + kDefaultRtt;
    } else {
      packet_result.receive_time = packets_received_times[i];
    }
    packet_results.push_back(packet_result);
  }
  return packet_results;
}

class MockUtilityFunction : public PccUtilityFunctionInterface {
 public:
  MOCK_METHOD(double,
              Compute,
              (const PccMonitorInterval& monitor_interval),
              (const, override));
};

}  // namespace

TEST(PccBitrateControllerTest, IncreaseRateWhenNoChangesForTestBitrates) {
  PccBitrateController bitrate_controller(
      kInitialConversionFactor, kInitialDynamicBoundary,
      kDynamicBoundaryIncrement, kDelayGradientCoefficient, kLossCoefficient,
      kThroughputCoefficient, kThroughputPower, kDelayGradientThreshold,
      kDelayGradientNegativeBound);
  VivaceUtilityFunction utility_function(
      kDelayGradientCoefficient, kLossCoefficient, kThroughputCoefficient,
      kThroughputPower, kDelayGradientThreshold, kDelayGradientNegativeBound);
  std::vector<PccMonitorInterval> monitor_block{
      PccMonitorInterval(kTargetSendingRate * (1 + kEpsilon), kStartTime,
                         kIntervalDuration),
      PccMonitorInterval(kTargetSendingRate * (1 - kEpsilon),
                         kStartTime + kIntervalDuration, kIntervalDuration)};
  monitor_block[0].OnPacketsFeedback(
      CreatePacketResults({kStartTime + kPacketsDelta,
                           kStartTime + kIntervalDuration + kPacketsDelta,
                           kStartTime + 3 * kIntervalDuration},
                          {}, {}));
  monitor_block[1].OnPacketsFeedback(
      CreatePacketResults({kStartTime + kPacketsDelta,
                           kStartTime + kIntervalDuration + kPacketsDelta,
                           kStartTime + 3 * kIntervalDuration},
                          {}, {}));
  // For both of the monitor intervals there were no change in rtt gradient
  // and in packet loss. Since the only difference is in the sending rate,
  // the higher sending rate should be chosen by congestion controller.
  EXPECT_GT(bitrate_controller
                .ComputeRateUpdateForOnlineLearningMode(monitor_block,
                                                        kTargetSendingRate)
                .bps(),
            kTargetSendingRate.bps());
}

TEST(PccBitrateControllerTest, NoChangesWhenUtilityFunctionDoesntChange) {
  std::unique_ptr<MockUtilityFunction> mock_utility_function =
      std::make_unique<MockUtilityFunction>();
  EXPECT_CALL(*mock_utility_function, Compute(::testing::_))
      .Times(2)
      .WillOnce(::testing::Return(100))
      .WillOnce(::testing::Return(100));

  PccBitrateController bitrate_controller(
      kInitialConversionFactor, kInitialDynamicBoundary,
      kDynamicBoundaryIncrement, std::move(mock_utility_function));
  std::vector<PccMonitorInterval> monitor_block{
      PccMonitorInterval(kTargetSendingRate * (1 + kEpsilon), kStartTime,
                         kIntervalDuration),
      PccMonitorInterval(kTargetSendingRate * (1 - kEpsilon),
                         kStartTime + kIntervalDuration, kIntervalDuration)};
  // To complete collecting feedback within monitor intervals.
  monitor_block[0].OnPacketsFeedback(
      CreatePacketResults({kStartTime + 3 * kIntervalDuration}, {}, {}));
  monitor_block[1].OnPacketsFeedback(
      CreatePacketResults({kStartTime + 3 * kIntervalDuration}, {}, {}));
  // Because we don't have any packets inside of monitor intervals, utility
  // function should be zero for both of them and the sending rate should not
  // change.
  EXPECT_EQ(bitrate_controller
                .ComputeRateUpdateForOnlineLearningMode(monitor_block,
                                                        kTargetSendingRate)
                .bps(),
            kTargetSendingRate.bps());
}

TEST(PccBitrateControllerTest, NoBoundaryWhenSmallGradient) {
  std::unique_ptr<MockUtilityFunction> mock_utility_function =
      std::make_unique<MockUtilityFunction>();
  constexpr double kFirstMonitorIntervalUtility = 0;
  const double kSecondMonitorIntervalUtility =
      2 * kTargetSendingRate.bps() * kEpsilon;

  EXPECT_CALL(*mock_utility_function, Compute(::testing::_))
      .Times(2)
      .WillOnce(::testing::Return(kFirstMonitorIntervalUtility))
      .WillOnce(::testing::Return(kSecondMonitorIntervalUtility));

  PccBitrateController bitrate_controller(
      kInitialConversionFactor, kInitialDynamicBoundary,
      kDynamicBoundaryIncrement, std::move(mock_utility_function));
  std::vector<PccMonitorInterval> monitor_block{
      PccMonitorInterval(kTargetSendingRate * (1 + kEpsilon), kStartTime,
                         kIntervalDuration),
      PccMonitorInterval(kTargetSendingRate * (1 - kEpsilon),
                         kStartTime + kIntervalDuration, kIntervalDuration)};
  // To complete collecting feedback within monitor intervals.
  monitor_block[0].OnPacketsFeedback(
      CreatePacketResults({kStartTime + 3 * kIntervalDuration}, {}, {}));
  monitor_block[1].OnPacketsFeedback(
      CreatePacketResults({kStartTime + 3 * kIntervalDuration}, {}, {}));

  double gradient =
      (kFirstMonitorIntervalUtility - kSecondMonitorIntervalUtility) /
      (kTargetSendingRate.bps() * 2 * kEpsilon);
  // When the gradient is small we don't hit the dynamic boundary.
  EXPECT_EQ(bitrate_controller
                .ComputeRateUpdateForOnlineLearningMode(monitor_block,
                                                        kTargetSendingRate)
                .bps(),
            kTargetSendingRate.bps() + kInitialConversionFactor * gradient);
}

TEST(PccBitrateControllerTest, FaceBoundaryWhenLargeGradient) {
  std::unique_ptr<MockUtilityFunction> mock_utility_function =
      std::make_unique<MockUtilityFunction>();
  constexpr double kFirstMonitorIntervalUtility = 0;
  const double kSecondMonitorIntervalUtility =
      10 * kInitialDynamicBoundary * kTargetSendingRate.bps() * 2 *
      kTargetSendingRate.bps() * kEpsilon;

  EXPECT_CALL(*mock_utility_function, Compute(::testing::_))
      .Times(4)
      .WillOnce(::testing::Return(kFirstMonitorIntervalUtility))
      .WillOnce(::testing::Return(kSecondMonitorIntervalUtility))
      .WillOnce(::testing::Return(kFirstMonitorIntervalUtility))
      .WillOnce(::testing::Return(kSecondMonitorIntervalUtility));

  PccBitrateController bitrate_controller(
      kInitialConversionFactor, kInitialDynamicBoundary,
      kDynamicBoundaryIncrement, std::move(mock_utility_function));
  std::vector<PccMonitorInterval> monitor_block{
      PccMonitorInterval(kTargetSendingRate * (1 + kEpsilon), kStartTime,
                         kIntervalDuration),
      PccMonitorInterval(kTargetSendingRate * (1 - kEpsilon),
                         kStartTime + kIntervalDuration, kIntervalDuration)};
  // To complete collecting feedback within monitor intervals.
  monitor_block[0].OnPacketsFeedback(
      CreatePacketResults({kStartTime + 3 * kIntervalDuration}, {}, {}));
  monitor_block[1].OnPacketsFeedback(
      CreatePacketResults({kStartTime + 3 * kIntervalDuration}, {}, {}));
  // The utility function gradient is too big and we hit the dynamic boundary.
  EXPECT_EQ(bitrate_controller.ComputeRateUpdateForOnlineLearningMode(
                monitor_block, kTargetSendingRate),
            kTargetSendingRate * (1 - kInitialDynamicBoundary));
  // For the second time we hit the dynamic boundary in the same direction, the
  // boundary should increase.
  EXPECT_EQ(bitrate_controller
                .ComputeRateUpdateForOnlineLearningMode(monitor_block,
                                                        kTargetSendingRate)
                .bps(),
            kTargetSendingRate.bps() *
                (1 - kInitialDynamicBoundary - kDynamicBoundaryIncrement));
}

TEST(PccBitrateControllerTest, SlowStartMode) {
  std::unique_ptr<MockUtilityFunction> mock_utility_function =
      std::make_unique<MockUtilityFunction>();
  constexpr double kFirstUtilityFunction = 1000;
  EXPECT_CALL(*mock_utility_function, Compute(::testing::_))
      .Times(4)
      // For first 3 calls we expect to stay in the SLOW_START mode and double
      // the sending rate since the utility function increases its value. For
      // the last call utility function decreases its value, this means that
      // we should not double the sending rate and exit SLOW_START mode.
      .WillOnce(::testing::Return(kFirstUtilityFunction))
      .WillOnce(::testing::Return(kFirstUtilityFunction + 1))
      .WillOnce(::testing::Return(kFirstUtilityFunction + 2))
      .WillOnce(::testing::Return(kFirstUtilityFunction + 1));

  PccBitrateController bitrate_controller(
      kInitialConversionFactor, kInitialDynamicBoundary,
      kDynamicBoundaryIncrement, std::move(mock_utility_function));
  std::vector<PccMonitorInterval> monitor_block{PccMonitorInterval(
      2 * kTargetSendingRate, kStartTime, kIntervalDuration)};
  // To complete collecting feedback within monitor intervals.
  monitor_block[0].OnPacketsFeedback(
      CreatePacketResults({kStartTime + 3 * kIntervalDuration}, {}, {}));
  EXPECT_EQ(
      bitrate_controller.ComputeRateUpdateForSlowStartMode(monitor_block[0]),
      kTargetSendingRate * 2);
  EXPECT_EQ(
      bitrate_controller.ComputeRateUpdateForSlowStartMode(monitor_block[0]),
      kTargetSendingRate * 2);
  EXPECT_EQ(
      bitrate_controller.ComputeRateUpdateForSlowStartMode(monitor_block[0]),
      kTargetSendingRate * 2);
  EXPECT_EQ(
      bitrate_controller.ComputeRateUpdateForSlowStartMode(monitor_block[0]),
      std::nullopt);
}

TEST(PccBitrateControllerTest, StepSizeIncrease) {
  std::unique_ptr<MockUtilityFunction> mock_utility_function =
      std::make_unique<MockUtilityFunction>();
  constexpr double kFirstMiUtilityFunction = 0;
  const double kSecondMiUtilityFunction =
      2 * kTargetSendingRate.bps() * kEpsilon;

  EXPECT_CALL(*mock_utility_function, Compute(::testing::_))
      .Times(4)
      .WillOnce(::testing::Return(kFirstMiUtilityFunction))
      .WillOnce(::testing::Return(kSecondMiUtilityFunction))
      .WillOnce(::testing::Return(kFirstMiUtilityFunction))
      .WillOnce(::testing::Return(kSecondMiUtilityFunction));
  std::vector<PccMonitorInterval> monitor_block{
      PccMonitorInterval(kTargetSendingRate * (1 + kEpsilon), kStartTime,
                         kIntervalDuration),
      PccMonitorInterval(kTargetSendingRate * (1 - kEpsilon),
                         kStartTime + kIntervalDuration, kIntervalDuration)};
  // To complete collecting feedback within monitor intervals.
  monitor_block[0].OnPacketsFeedback(
      CreatePacketResults({kStartTime + 3 * kIntervalDuration}, {}, {}));
  monitor_block[1].OnPacketsFeedback(
      CreatePacketResults({kStartTime + 3 * kIntervalDuration}, {}, {}));

  double gradient = (kFirstMiUtilityFunction - kSecondMiUtilityFunction) /
                    (kTargetSendingRate.bps() * 2 * kEpsilon);
  PccBitrateController bitrate_controller(
      kInitialConversionFactor, kInitialDynamicBoundary,
      kDynamicBoundaryIncrement, std::move(mock_utility_function));
  // If we are moving in the same direction - the step size should increase.
  EXPECT_EQ(bitrate_controller
                .ComputeRateUpdateForOnlineLearningMode(monitor_block,
                                                        kTargetSendingRate)
                .bps(),
            kTargetSendingRate.bps() + kInitialConversionFactor * gradient);
  EXPECT_EQ(bitrate_controller
                .ComputeRateUpdateForOnlineLearningMode(monitor_block,
                                                        kTargetSendingRate)
                .bps(),
            kTargetSendingRate.bps() + 2 * kInitialConversionFactor * gradient);
}

}  // namespace test
}  // namespace pcc
}  // namespace webrtc
