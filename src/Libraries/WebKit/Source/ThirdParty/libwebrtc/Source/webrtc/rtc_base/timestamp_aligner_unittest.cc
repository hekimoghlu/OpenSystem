/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
#include "rtc_base/timestamp_aligner.h"

#include <math.h>

#include <algorithm>
#include <limits>

#include "rtc_base/random.h"
#include "rtc_base/time_utils.h"
#include "test/gtest.h"

namespace rtc {

namespace {
// Computes the difference x_k - mean(x), when x_k is the linear sequence x_k =
// k, and the "mean" is plain mean for the first `window_size` samples, followed
// by exponential averaging with weight 1 / `window_size` for each new sample.
// This is needed to predict the effect of camera clock drift on the timestamp
// translation. See the comment on TimestampAligner::UpdateOffset for more
// context.
double MeanTimeDifference(int nsamples, int window_size) {
  if (nsamples <= window_size) {
    // Plain averaging.
    return nsamples / 2.0;
  } else {
    // Exponential convergence towards
    // interval_error * (window_size - 1)
    double alpha = 1.0 - 1.0 / window_size;

    return ((window_size - 1) -
            (window_size / 2.0 - 1) * pow(alpha, nsamples - window_size));
  }
}

class TimestampAlignerForTest : public TimestampAligner {
  // Make internal methods accessible to testing.
 public:
  using TimestampAligner::ClipTimestamp;
  using TimestampAligner::UpdateOffset;
};

void TestTimestampFilter(double rel_freq_error) {
  TimestampAlignerForTest timestamp_aligner_for_test;
  TimestampAligner timestamp_aligner;
  const int64_t kEpoch = 10000;
  const int64_t kJitterUs = 5000;
  const int64_t kIntervalUs = 33333;  // 30 FPS
  const int kWindowSize = 100;
  const int kNumFrames = 3 * kWindowSize;

  int64_t interval_error_us = kIntervalUs * rel_freq_error;
  int64_t system_start_us = rtc::TimeMicros();
  webrtc::Random random(17);

  int64_t prev_translated_time_us = system_start_us;

  for (int i = 0; i < kNumFrames; i++) {
    // Camera time subject to drift.
    int64_t camera_time_us = kEpoch + i * (kIntervalUs + interval_error_us);
    int64_t system_time_us = system_start_us + i * kIntervalUs;
    // And system time readings are subject to jitter.
    int64_t system_measured_us = system_time_us + random.Rand(kJitterUs);

    int64_t offset_us = timestamp_aligner_for_test.UpdateOffset(
        camera_time_us, system_measured_us);

    int64_t filtered_time_us = camera_time_us + offset_us;
    int64_t translated_time_us = timestamp_aligner_for_test.ClipTimestamp(
        filtered_time_us, system_measured_us);

    // Check that we get identical result from the all-in-one helper method.
    ASSERT_EQ(translated_time_us, timestamp_aligner.TranslateTimestamp(
                                      camera_time_us, system_measured_us));

    EXPECT_LE(translated_time_us, system_measured_us);
    EXPECT_GE(translated_time_us,
              prev_translated_time_us + rtc::kNumMicrosecsPerMillisec);

    // The relative frequency error contributes to the expected error
    // by a factor which is the difference between the current time
    // and the average of earlier sample times.
    int64_t expected_error_us =
        kJitterUs / 2 +
        rel_freq_error * kIntervalUs * MeanTimeDifference(i, kWindowSize);

    int64_t bias_us = filtered_time_us - translated_time_us;
    EXPECT_GE(bias_us, 0);

    if (i == 0) {
      EXPECT_EQ(translated_time_us, system_measured_us);
    } else {
      EXPECT_NEAR(filtered_time_us, system_time_us + expected_error_us,
                  2.0 * kJitterUs / sqrt(std::max(i, kWindowSize)));
    }
    // If the camera clock runs too fast (rel_freq_error > 0.0), The
    // bias is expected to roughly cancel the expected error from the
    // clock drift, as this grows. Otherwise, it reflects the
    // measurement noise. The tolerances here were selected after some
    // trial and error.
    if (i < 10 || rel_freq_error <= 0.0) {
      EXPECT_LE(bias_us, 3000);
    } else {
      EXPECT_NEAR(bias_us, expected_error_us, 1500);
    }
    prev_translated_time_us = translated_time_us;
  }
}

}  // Anonymous namespace

TEST(TimestampAlignerTest, AttenuateTimestampJitterNoDrift) {
  TestTimestampFilter(0.0);
}

// 100 ppm is a worst case for a reasonable crystal.
TEST(TimestampAlignerTest, AttenuateTimestampJitterSmallPosDrift) {
  TestTimestampFilter(0.0001);
}

TEST(TimestampAlignerTest, AttenuateTimestampJitterSmallNegDrift) {
  TestTimestampFilter(-0.0001);
}

// 3000 ppm, 3 ms / s, is the worst observed drift, see
// https://bugs.chromium.org/p/webrtc/issues/detail?id=5456
TEST(TimestampAlignerTest, AttenuateTimestampJitterLargePosDrift) {
  TestTimestampFilter(0.003);
}

TEST(TimestampAlignerTest, AttenuateTimestampJitterLargeNegDrift) {
  TestTimestampFilter(-0.003);
}

// Exhibits a mostly hypothetical problem, where certain inputs to the
// TimestampAligner.UpdateOffset filter result in non-monotonous
// translated timestamps. This test verifies that the ClipTimestamp
// logic handles this case correctly.
TEST(TimestampAlignerTest, ClipToMonotonous) {
  TimestampAlignerForTest timestamp_aligner;

  // For system time stamps { 0, s1, s1 + s2 }, and camera timestamps
  // {0, c1, c1 + c2}, we exhibit non-monotonous behaviour if and only
  // if c1 > s1 + 2 s2 + 4 c2.
  const int kNumSamples = 3;
  const int64_t kCaptureTimeUs[kNumSamples] = {0, 80000, 90001};
  const int64_t kSystemTimeUs[kNumSamples] = {0, 10000, 20000};
  const int64_t expected_offset_us[kNumSamples] = {0, -35000, -46667};

  // Non-monotonic translated timestamps can happen when only for
  // translated timestamps in the future. Which is tolerated if
  // `timestamp_aligner.clip_bias_us` is large enough. Instead of
  // changing that private member for this test, just add the bias to
  // `kSystemTimeUs` when calling ClipTimestamp.
  const int64_t kClipBiasUs = 100000;

  bool did_clip = false;
  int64_t prev_timestamp_us = std::numeric_limits<int64_t>::min();
  for (int i = 0; i < kNumSamples; i++) {
    int64_t offset_us =
        timestamp_aligner.UpdateOffset(kCaptureTimeUs[i], kSystemTimeUs[i]);
    EXPECT_EQ(offset_us, expected_offset_us[i]);

    int64_t translated_timestamp_us = kCaptureTimeUs[i] + offset_us;
    int64_t clip_timestamp_us = timestamp_aligner.ClipTimestamp(
        translated_timestamp_us, kSystemTimeUs[i] + kClipBiasUs);
    if (translated_timestamp_us <= prev_timestamp_us) {
      did_clip = true;
      EXPECT_EQ(clip_timestamp_us,
                prev_timestamp_us + rtc::kNumMicrosecsPerMillisec);
    } else {
      // No change from clipping.
      EXPECT_EQ(clip_timestamp_us, translated_timestamp_us);
    }
    prev_timestamp_us = clip_timestamp_us;
  }
  EXPECT_TRUE(did_clip);
}

TEST(TimestampAlignerTest, TranslateTimestampWithoutStateUpdate) {
  TimestampAligner timestamp_aligner;

  constexpr int kNumSamples = 4;
  constexpr int64_t kCaptureTimeUs[kNumSamples] = {0, 80000, 90001, 100000};
  constexpr int64_t kSystemTimeUs[kNumSamples] = {0, 10000, 20000, 30000};
  constexpr int64_t kQueryCaptureTimeOffsetUs[kNumSamples] = {0, 123, -321,
                                                              345};

  for (int i = 0; i < kNumSamples; i++) {
    int64_t reference_timestamp = timestamp_aligner.TranslateTimestamp(
        kCaptureTimeUs[i], kSystemTimeUs[i]);
    EXPECT_EQ(reference_timestamp - kQueryCaptureTimeOffsetUs[i],
              timestamp_aligner.TranslateTimestamp(
                  kCaptureTimeUs[i] - kQueryCaptureTimeOffsetUs[i]));
  }
}

}  // namespace rtc
