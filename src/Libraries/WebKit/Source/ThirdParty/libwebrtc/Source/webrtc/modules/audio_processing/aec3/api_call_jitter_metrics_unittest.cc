/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#include "modules/audio_processing/aec3/api_call_jitter_metrics.h"

#include "modules/audio_processing/aec3/aec3_common.h"
#include "test/gtest.h"

namespace webrtc {

// Verify constant jitter.
TEST(ApiCallJitterMetrics, ConstantJitter) {
  for (int jitter = 1; jitter < 20; ++jitter) {
    ApiCallJitterMetrics metrics;
    for (size_t k = 0; k < 30 * kNumBlocksPerSecond; ++k) {
      for (int j = 0; j < jitter; ++j) {
        metrics.ReportRenderCall();
      }

      for (int j = 0; j < jitter; ++j) {
        metrics.ReportCaptureCall();

        if (metrics.WillReportMetricsAtNextCapture()) {
          EXPECT_EQ(jitter, metrics.render_jitter().min());
          EXPECT_EQ(jitter, metrics.render_jitter().max());
          EXPECT_EQ(jitter, metrics.capture_jitter().min());
          EXPECT_EQ(jitter, metrics.capture_jitter().max());
        }
      }
    }
  }
}

// Verify peaky jitter for the render.
TEST(ApiCallJitterMetrics, JitterPeakRender) {
  constexpr int kMinJitter = 2;
  constexpr int kJitterPeak = 10;
  constexpr int kPeakInterval = 100;

  ApiCallJitterMetrics metrics;
  int render_surplus = 0;

  for (size_t k = 0; k < 30 * kNumBlocksPerSecond; ++k) {
    const int num_render_calls =
        k % kPeakInterval == 0 ? kJitterPeak : kMinJitter;
    for (int j = 0; j < num_render_calls; ++j) {
      metrics.ReportRenderCall();
      ++render_surplus;
    }

    ASSERT_LE(kMinJitter, render_surplus);
    const int num_capture_calls =
        render_surplus == kMinJitter ? kMinJitter : kMinJitter + 1;
    for (int j = 0; j < num_capture_calls; ++j) {
      metrics.ReportCaptureCall();

      if (metrics.WillReportMetricsAtNextCapture()) {
        EXPECT_EQ(kMinJitter, metrics.render_jitter().min());
        EXPECT_EQ(kJitterPeak, metrics.render_jitter().max());
        EXPECT_EQ(kMinJitter, metrics.capture_jitter().min());
        EXPECT_EQ(kMinJitter + 1, metrics.capture_jitter().max());
      }
      --render_surplus;
    }
  }
}

// Verify peaky jitter for the capture.
TEST(ApiCallJitterMetrics, JitterPeakCapture) {
  constexpr int kMinJitter = 2;
  constexpr int kJitterPeak = 10;
  constexpr int kPeakInterval = 100;

  ApiCallJitterMetrics metrics;
  int capture_surplus = kMinJitter;

  for (size_t k = 0; k < 30 * kNumBlocksPerSecond; ++k) {
    ASSERT_LE(kMinJitter, capture_surplus);
    const int num_render_calls =
        capture_surplus == kMinJitter ? kMinJitter : kMinJitter + 1;
    for (int j = 0; j < num_render_calls; ++j) {
      metrics.ReportRenderCall();
      --capture_surplus;
    }

    const int num_capture_calls =
        k % kPeakInterval == 0 ? kJitterPeak : kMinJitter;
    for (int j = 0; j < num_capture_calls; ++j) {
      metrics.ReportCaptureCall();

      if (metrics.WillReportMetricsAtNextCapture()) {
        EXPECT_EQ(kMinJitter, metrics.render_jitter().min());
        EXPECT_EQ(kMinJitter + 1, metrics.render_jitter().max());
        EXPECT_EQ(kMinJitter, metrics.capture_jitter().min());
        EXPECT_EQ(kJitterPeak, metrics.capture_jitter().max());
      }
      ++capture_surplus;
    }
  }
}

}  // namespace webrtc
