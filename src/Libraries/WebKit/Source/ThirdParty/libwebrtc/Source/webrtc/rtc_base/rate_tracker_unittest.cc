/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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
#include "rtc_base/rate_tracker.h"

#include "test/gtest.h"

namespace rtc {
namespace {
const uint32_t kBucketIntervalMs = 100;
}  // namespace

class RateTrackerForTest : public RateTracker {
 public:
  RateTrackerForTest() : RateTracker(kBucketIntervalMs, 10u), time_(0) {}
  int64_t Time() const override { return time_; }
  void AdvanceTime(int delta) { time_ += delta; }

 private:
  int64_t time_;
};

TEST(RateTrackerTest, Test30FPS) {
  RateTrackerForTest tracker;

  for (int i = 0; i < 300; ++i) {
    tracker.AddSamples(1);
    tracker.AdvanceTime(33);
    if (i % 3 == 0) {
      tracker.AdvanceTime(1);
    }
  }
  EXPECT_DOUBLE_EQ(30.0, tracker.ComputeRateForInterval(50000));
}

TEST(RateTrackerTest, Test60FPS) {
  RateTrackerForTest tracker;

  for (int i = 0; i < 300; ++i) {
    tracker.AddSamples(1);
    tracker.AdvanceTime(16);
    if (i % 3 != 0) {
      tracker.AdvanceTime(1);
    }
  }
  EXPECT_DOUBLE_EQ(60.0, tracker.ComputeRateForInterval(1000));
}

TEST(RateTrackerTest, TestRateTrackerBasics) {
  RateTrackerForTest tracker;
  EXPECT_DOUBLE_EQ(0.0, tracker.ComputeRateForInterval(1000));

  // Add a sample.
  tracker.AddSamples(1234);
  // Advance the clock by less than one bucket interval (no rate returned).
  tracker.AdvanceTime(kBucketIntervalMs - 1);
  EXPECT_DOUBLE_EQ(0.0, tracker.ComputeRate());
  // Advance the clock by 100 ms (one bucket interval).
  tracker.AdvanceTime(1);
  EXPECT_DOUBLE_EQ(12340.0, tracker.ComputeRateForInterval(1000));
  EXPECT_DOUBLE_EQ(12340.0, tracker.ComputeRate());
  EXPECT_EQ(1234U, tracker.TotalSampleCount());
  EXPECT_DOUBLE_EQ(12340.0, tracker.ComputeTotalRate());

  // Repeat.
  tracker.AddSamples(1234);
  tracker.AdvanceTime(100);
  EXPECT_DOUBLE_EQ(12340.0, tracker.ComputeRateForInterval(1000));
  EXPECT_DOUBLE_EQ(12340.0, tracker.ComputeRate());
  EXPECT_EQ(1234U * 2, tracker.TotalSampleCount());
  EXPECT_DOUBLE_EQ(12340.0, tracker.ComputeTotalRate());

  // Advance the clock by 800 ms, so we've elapsed a full second.
  // units_second should now be filled in properly.
  tracker.AdvanceTime(800);
  EXPECT_DOUBLE_EQ(1234.0 * 2.0, tracker.ComputeRateForInterval(1000));
  EXPECT_DOUBLE_EQ(1234.0 * 2.0, tracker.ComputeRate());
  EXPECT_EQ(1234U * 2, tracker.TotalSampleCount());
  EXPECT_DOUBLE_EQ(1234.0 * 2.0, tracker.ComputeTotalRate());

  // Poll the tracker again immediately. The reported rate should stay the same.
  EXPECT_DOUBLE_EQ(1234.0 * 2.0, tracker.ComputeRateForInterval(1000));
  EXPECT_DOUBLE_EQ(1234.0 * 2.0, tracker.ComputeRate());
  EXPECT_EQ(1234U * 2, tracker.TotalSampleCount());
  EXPECT_DOUBLE_EQ(1234.0 * 2.0, tracker.ComputeTotalRate());

  // Do nothing and advance by a second. We should drop down to zero.
  tracker.AdvanceTime(1000);
  EXPECT_DOUBLE_EQ(0.0, tracker.ComputeRateForInterval(1000));
  EXPECT_DOUBLE_EQ(0.0, tracker.ComputeRate());
  EXPECT_EQ(1234U * 2, tracker.TotalSampleCount());
  EXPECT_DOUBLE_EQ(1234.0, tracker.ComputeTotalRate());

  // Send a bunch of data at a constant rate for 5.5 "seconds".
  // We should report the rate properly.
  for (int i = 0; i < 5500; i += 100) {
    tracker.AddSamples(9876U);
    tracker.AdvanceTime(100);
  }
  EXPECT_DOUBLE_EQ(9876.0 * 10.0, tracker.ComputeRateForInterval(1000));
  EXPECT_DOUBLE_EQ(9876.0 * 10.0, tracker.ComputeRate());
  EXPECT_EQ(1234U * 2 + 9876U * 55, tracker.TotalSampleCount());
  EXPECT_DOUBLE_EQ((1234.0 * 2.0 + 9876.0 * 55.0) / 7.5,
                   tracker.ComputeTotalRate());

  // Advance the clock by 500 ms. Since we sent nothing over this half-second,
  // the reported rate should be reduced by half.
  tracker.AdvanceTime(500);
  EXPECT_DOUBLE_EQ(9876.0 * 5.0, tracker.ComputeRateForInterval(1000));
  EXPECT_DOUBLE_EQ(9876.0 * 5.0, tracker.ComputeRate());
  EXPECT_EQ(1234U * 2 + 9876U * 55, tracker.TotalSampleCount());
  EXPECT_DOUBLE_EQ((1234.0 * 2.0 + 9876.0 * 55.0) / 8.0,
                   tracker.ComputeTotalRate());

  // Rate over the last half second should be zero.
  EXPECT_DOUBLE_EQ(0.0, tracker.ComputeRateForInterval(500));
}

TEST(RateTrackerTest, TestLongPeriodBetweenSamples) {
  RateTrackerForTest tracker;
  tracker.AddSamples(1);
  tracker.AdvanceTime(1000);
  EXPECT_DOUBLE_EQ(1.0, tracker.ComputeRate());

  tracker.AdvanceTime(2000);
  EXPECT_DOUBLE_EQ(0.0, tracker.ComputeRate());

  tracker.AdvanceTime(2000);
  tracker.AddSamples(1);
  EXPECT_DOUBLE_EQ(1.0, tracker.ComputeRate());
}

TEST(RateTrackerTest, TestRolloff) {
  RateTrackerForTest tracker;
  for (int i = 0; i < 10; ++i) {
    tracker.AddSamples(1U);
    tracker.AdvanceTime(100);
  }
  EXPECT_DOUBLE_EQ(10.0, tracker.ComputeRate());

  for (int i = 0; i < 10; ++i) {
    tracker.AddSamples(1U);
    tracker.AdvanceTime(50);
  }
  EXPECT_DOUBLE_EQ(15.0, tracker.ComputeRate());
  EXPECT_DOUBLE_EQ(20.0, tracker.ComputeRateForInterval(500));

  for (int i = 0; i < 10; ++i) {
    tracker.AddSamples(1U);
    tracker.AdvanceTime(50);
  }
  EXPECT_DOUBLE_EQ(20.0, tracker.ComputeRate());
}

TEST(RateTrackerTest, TestGetUnitSecondsAfterInitialValue) {
  RateTrackerForTest tracker;
  tracker.AddSamples(1234);
  tracker.AdvanceTime(1000);
  EXPECT_DOUBLE_EQ(1234.0, tracker.ComputeRateForInterval(1000));
}

TEST(RateTrackerTest, TestLargeNumbers) {
  RateTrackerForTest tracker;
  const uint64_t large_number = 0x100000000;
  tracker.AddSamples(large_number);
  tracker.AdvanceTime(1000);
  tracker.AddSamples(large_number);
  EXPECT_DOUBLE_EQ(large_number * 2, tracker.ComputeRate());
}

}  // namespace rtc
