/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
#include "modules/rtp_rtcp/source/capture_clock_offset_updater.h"

#include "system_wrappers/include/ntp_time.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

TEST(AbsoluteCaptureTimeReceiverTest,
     SkipEstimatedCaptureClockOffsetIfRemoteToLocalClockOffsetIsUnknown) {
  static const std::optional<int64_t> kRemoteCaptureClockOffset =
      Int64MsToQ32x32(-350);
  CaptureClockOffsetUpdater updater;
  updater.SetRemoteToLocalClockOffset(std::nullopt);
  EXPECT_EQ(
      updater.AdjustEstimatedCaptureClockOffset(kRemoteCaptureClockOffset),
      std::nullopt);
}

TEST(AbsoluteCaptureTimeReceiverTest,
     SkipEstimatedCaptureClockOffsetIfRemoteCaptureClockOffsetIsUnknown) {
  static const std::optional<int64_t> kCaptureClockOffsetNull = std::nullopt;
  CaptureClockOffsetUpdater updater;
  updater.SetRemoteToLocalClockOffset(0);
  EXPECT_EQ(updater.AdjustEstimatedCaptureClockOffset(kCaptureClockOffsetNull),
            kCaptureClockOffsetNull);

  static const std::optional<int64_t> kRemoteCaptureClockOffset =
      Int64MsToQ32x32(-350);
  EXPECT_EQ(
      updater.AdjustEstimatedCaptureClockOffset(kRemoteCaptureClockOffset),
      kRemoteCaptureClockOffset);
}

TEST(AbsoluteCaptureTimeReceiverTest, EstimatedCaptureClockOffsetArithmetic) {
  static const std::optional<int64_t> kRemoteCaptureClockOffset =
      Int64MsToQ32x32(-350);
  static const std::optional<int64_t> kRemoteToLocalClockOffset =
      Int64MsToQ32x32(-7000007);
  CaptureClockOffsetUpdater updater;
  updater.SetRemoteToLocalClockOffset(kRemoteToLocalClockOffset);
  EXPECT_THAT(
      updater.AdjustEstimatedCaptureClockOffset(kRemoteCaptureClockOffset),
      ::testing::Optional(::testing::Eq(*kRemoteCaptureClockOffset +
                                        *kRemoteToLocalClockOffset)));
}

TEST(AbsoluteCaptureTimeReceiverTest, ConvertClockOffset) {
  constexpr TimeDelta kNegative = TimeDelta::Millis(-350);
  constexpr int64_t kNegativeQ32x32 =
      kNegative.ms() * (NtpTime::kFractionsPerSecond / 1000);
  constexpr TimeDelta kPositive = TimeDelta::Millis(400);
  constexpr int64_t kPositiveQ32x32 =
      kPositive.ms() * (NtpTime::kFractionsPerSecond / 1000);
  constexpr TimeDelta kEpsilon = TimeDelta::Millis(1);
  std::optional<TimeDelta> converted =
      CaptureClockOffsetUpdater::ConvertsToTimeDela(kNegativeQ32x32);
  EXPECT_GT(converted, kNegative - kEpsilon);
  EXPECT_LT(converted, kNegative + kEpsilon);

  converted = CaptureClockOffsetUpdater::ConvertsToTimeDela(kPositiveQ32x32);
  EXPECT_GT(converted, kPositive - kEpsilon);
  EXPECT_LT(converted, kPositive + kEpsilon);

  EXPECT_FALSE(
      CaptureClockOffsetUpdater::ConvertsToTimeDela(std::nullopt).has_value());
}

}  // namespace webrtc
