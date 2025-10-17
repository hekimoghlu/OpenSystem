/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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
#include "rtc_base/frequency_tracker.h"

#include <cstdlib>
#include <limits>
#include <optional>

#include "api/units/frequency.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::AllOf;
using ::testing::Gt;
using ::testing::Lt;

constexpr TimeDelta kWindow = TimeDelta::Millis(500);
constexpr TimeDelta kEpsilon = TimeDelta::Millis(1);

TEST(FrequencyTrackerTest, ReturnsNulloptInitially) {
  Timestamp now = Timestamp::Seconds(12'345);
  FrequencyTracker stats(kWindow);

  EXPECT_EQ(stats.Rate(now), std::nullopt);
}

TEST(FrequencyTrackerTest, ReturnsNulloptAfterSingleDataPoint) {
  Timestamp now = Timestamp::Seconds(12'345);
  FrequencyTracker stats(kWindow);

  stats.Update(now);
  now += TimeDelta::Millis(10);

  EXPECT_EQ(stats.Rate(now), std::nullopt);
}

TEST(FrequencyTrackerTest, ReturnsRateAfterTwoMeasurements) {
  Timestamp now = Timestamp::Seconds(12'345);
  FrequencyTracker stats(kWindow);

  stats.Update(now);
  now += TimeDelta::Millis(1);
  stats.Update(now);

  // 1 event per 1 ms ~= 1'000 events per second.
  EXPECT_EQ(stats.Rate(now), Frequency::Hertz(1'000));
}

TEST(FrequencyTrackerTest, MeasuresConstantRate) {
  const Timestamp start = Timestamp::Seconds(12'345);
  const TimeDelta kInterval = TimeDelta::Millis(10);
  const Frequency kConstantRate = 1 / kInterval;

  Timestamp now = start;
  FrequencyTracker stats(kWindow);

  stats.Update(now);
  Frequency last_error = Frequency::PlusInfinity();
  for (TimeDelta i = TimeDelta::Zero(); i < kWindow; i += kInterval) {
    SCOPED_TRACE(ToString(i));
    now += kInterval;
    stats.Update(now);

    // Until window is full, rate is measured over a smaller window and might
    // look larger than the constant rate.
    std::optional<Frequency> rate = stats.Rate(now);
    ASSERT_GE(rate, kConstantRate);

    // Expect the estimation error to decrease as the window is extended.
    Frequency error = *rate - kConstantRate;
    EXPECT_LE(error, last_error);
    last_error = error;
  }

  // Once window is full, rate measurment should be stable.
  for (TimeDelta i = TimeDelta::Zero(); i < kInterval;
       i += TimeDelta::Millis(1)) {
    SCOPED_TRACE(ToString(i));
    EXPECT_EQ(stats.Rate(now + i), kConstantRate);
  }
}

TEST(FrequencyTrackerTest, CanMeasureFractionalRate) {
  const TimeDelta kInterval = TimeDelta::Millis(134);
  Timestamp now = Timestamp::Seconds(12'345);
  // FrequencyTracker counts number of events in the window, thus when window is
  // fraction of 1 second, number of events per second would always be integer.
  const TimeDelta window = TimeDelta::Seconds(2);

  FrequencyTracker framerate(window);
  framerate.Update(now);
  for (TimeDelta i = TimeDelta::Zero(); i < window; i += kInterval) {
    now += kInterval;
    framerate.Update(now);
  }

  // Should be aproximitly 7.5 fps
  EXPECT_THAT(framerate.Rate(now),
              AllOf(Gt(Frequency::Hertz(7)), Lt(Frequency::Hertz(8))));
}

TEST(FrequencyTrackerTest, IncreasingThenDecreasingRate) {
  const int64_t kLargeSize = 1'500;
  const int64_t kSmallSize = 300;
  const TimeDelta kLargeInterval = TimeDelta::Millis(10);
  const TimeDelta kSmallInterval = TimeDelta::Millis(2);

  Timestamp now = Timestamp::Seconds(12'345);
  FrequencyTracker stats(kWindow);

  stats.Update(kLargeSize, now);
  for (TimeDelta i = TimeDelta::Zero(); i < kWindow; i += kLargeInterval) {
    SCOPED_TRACE(ToString(i));
    now += kLargeInterval;
    stats.Update(kLargeSize, now);
  }
  std::optional<Frequency> last_rate = stats.Rate(now);
  EXPECT_EQ(last_rate, kLargeSize / kLargeInterval);

  // Decrease rate with smaller measurments.
  for (TimeDelta i = TimeDelta::Zero(); i < kWindow; i += kLargeInterval) {
    SCOPED_TRACE(ToString(i));
    now += kLargeInterval;
    stats.Update(kSmallSize, now);

    std::optional<Frequency> rate = stats.Rate(now);
    EXPECT_LT(rate, last_rate);

    last_rate = rate;
  }
  EXPECT_EQ(last_rate, kSmallSize / kLargeInterval);

  // Increase rate with more frequent measurments.
  for (TimeDelta i = TimeDelta::Zero(); i < kWindow; i += kSmallInterval) {
    SCOPED_TRACE(ToString(i));
    now += kSmallInterval;
    stats.Update(kSmallSize, now);

    std::optional<Frequency> rate = stats.Rate(now);
    EXPECT_GE(rate, last_rate);

    last_rate = rate;
  }
  EXPECT_EQ(last_rate, kSmallSize / kSmallInterval);
}

TEST(FrequencyTrackerTest, ResetAfterSilence) {
  const TimeDelta kInterval = TimeDelta::Millis(10);
  const int64_t kPixels = 640 * 360;

  Timestamp now = Timestamp::Seconds(12'345);
  FrequencyTracker pixel_rate(kWindow);

  // Feed data until window has been filled.
  pixel_rate.Update(kPixels, now);
  for (TimeDelta i = TimeDelta::Zero(); i < kWindow; i += kInterval) {
    now += kInterval;
    pixel_rate.Update(kPixels, now);
  }
  ASSERT_GT(pixel_rate.Rate(now), Frequency::Zero());

  now += kWindow + kEpsilon;
  // Silence over window size should trigger auto reset for coming sample.
  EXPECT_EQ(pixel_rate.Rate(now), std::nullopt);
  pixel_rate.Update(kPixels, now);
  // Single measurment after reset is not enough to estimate the rate.
  EXPECT_EQ(pixel_rate.Rate(now), std::nullopt);

  // Manual reset, add the same check again.
  pixel_rate.Reset();
  EXPECT_EQ(pixel_rate.Rate(now), std::nullopt);
  now += kInterval;
  pixel_rate.Update(kPixels, now);
  EXPECT_EQ(pixel_rate.Rate(now), std::nullopt);
}

TEST(FrequencyTrackerTest, ReturnsNulloptWhenOverflows) {
  Timestamp now = Timestamp::Seconds(12'345);
  FrequencyTracker stats(kWindow);

  int64_t very_large_number = std::numeric_limits<int64_t>::max();
  stats.Update(very_large_number, now);
  now += kEpsilon;
  stats.Update(very_large_number, now);

  EXPECT_EQ(stats.Rate(now), std::nullopt);
}

}  // namespace
}  // namespace webrtc
