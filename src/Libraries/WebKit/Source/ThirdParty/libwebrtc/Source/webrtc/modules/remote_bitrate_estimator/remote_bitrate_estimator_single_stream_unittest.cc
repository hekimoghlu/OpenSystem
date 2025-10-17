/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#include "modules/remote_bitrate_estimator/remote_bitrate_estimator_single_stream.h"

#include <memory>

#include "api/environment/environment_factory.h"
#include "modules/remote_bitrate_estimator/remote_bitrate_estimator_unittest_helper.h"
#include "test/gtest.h"

namespace webrtc {

class RemoteBitrateEstimatorSingleTest : public RemoteBitrateEstimatorTest {
 public:
  RemoteBitrateEstimatorSingleTest() {}

  RemoteBitrateEstimatorSingleTest(const RemoteBitrateEstimatorSingleTest&) =
      delete;
  RemoteBitrateEstimatorSingleTest& operator=(
      const RemoteBitrateEstimatorSingleTest&) = delete;

  void SetUp() override {
    bitrate_estimator_ = std::make_unique<RemoteBitrateEstimatorSingleStream>(
        CreateEnvironment(&clock_), bitrate_observer_.get());
  }
};

TEST_F(RemoteBitrateEstimatorSingleTest, InitialBehavior) {
  InitialBehaviorTestHelper(508017);
}

TEST_F(RemoteBitrateEstimatorSingleTest, RateIncreaseReordering) {
  RateIncreaseReorderingTestHelper(506422);
}

TEST_F(RemoteBitrateEstimatorSingleTest, RateIncreaseRtpTimestamps) {
  RateIncreaseRtpTimestampsTestHelper(1267);
}

TEST_F(RemoteBitrateEstimatorSingleTest, CapacityDropOneStream) {
  CapacityDropTestHelper(1, false, 633, 0);
}

TEST_F(RemoteBitrateEstimatorSingleTest, CapacityDropOneStreamWrap) {
  CapacityDropTestHelper(1, true, 633, 0);
}

TEST_F(RemoteBitrateEstimatorSingleTest, CapacityDropTwoStreamsWrap) {
  CapacityDropTestHelper(2, true, 567, 0);
}

TEST_F(RemoteBitrateEstimatorSingleTest, CapacityDropThreeStreamsWrap) {
  CapacityDropTestHelper(3, true, 567, 0);
}

TEST_F(RemoteBitrateEstimatorSingleTest, CapacityDropThirteenStreamsWrap) {
  CapacityDropTestHelper(13, true, 767, 0);
}

TEST_F(RemoteBitrateEstimatorSingleTest, CapacityDropNineteenStreamsWrap) {
  CapacityDropTestHelper(19, true, 767, 0);
}

TEST_F(RemoteBitrateEstimatorSingleTest, CapacityDropThirtyStreamsWrap) {
  CapacityDropTestHelper(30, true, 733, 0);
}

TEST_F(RemoteBitrateEstimatorSingleTest, TestTimestampGrouping) {
  TestTimestampGroupingTestHelper();
}
}  // namespace webrtc
