/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
#include "modules/audio_processing/aec3/render_delay_controller_metrics.h"

#include <optional>

#include "modules/audio_processing/aec3/aec3_common.h"
#include "system_wrappers/include/metrics.h"
#include "test/gtest.h"

namespace webrtc {

// Verify the general functionality of RenderDelayControllerMetrics.
TEST(RenderDelayControllerMetrics, NormalUsage) {
  metrics::Reset();

  RenderDelayControllerMetrics metrics;

  int expected_num_metric_reports = 0;

  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < kMetricsReportingIntervalBlocks - 1; ++k) {
      metrics.Update(std::nullopt, std::nullopt,
                     ClockdriftDetector::Level::kNone);
    }
    EXPECT_METRIC_EQ(
        metrics::NumSamples("WebRTC.Audio.EchoCanceller.EchoPathDelay"),
        expected_num_metric_reports);
    EXPECT_METRIC_EQ(
        metrics::NumSamples("WebRTC.Audio.EchoCanceller.BufferDelay"),
        expected_num_metric_reports);
    EXPECT_METRIC_EQ(metrics::NumSamples(
                         "WebRTC.Audio.EchoCanceller.ReliableDelayEstimates"),
                     expected_num_metric_reports);
    EXPECT_METRIC_EQ(
        metrics::NumSamples("WebRTC.Audio.EchoCanceller.DelayChanges"),
        expected_num_metric_reports);
    EXPECT_METRIC_EQ(
        metrics::NumSamples("WebRTC.Audio.EchoCanceller.Clockdrift"),
        expected_num_metric_reports);

    // We expect metric reports every kMetricsReportingIntervalBlocks blocks.
    ++expected_num_metric_reports;

    metrics.Update(std::nullopt, std::nullopt,
                   ClockdriftDetector::Level::kNone);
    EXPECT_METRIC_EQ(
        metrics::NumSamples("WebRTC.Audio.EchoCanceller.EchoPathDelay"),
        expected_num_metric_reports);
    EXPECT_METRIC_EQ(
        metrics::NumSamples("WebRTC.Audio.EchoCanceller.BufferDelay"),
        expected_num_metric_reports);
    EXPECT_METRIC_EQ(metrics::NumSamples(
                         "WebRTC.Audio.EchoCanceller.ReliableDelayEstimates"),
                     expected_num_metric_reports);
    EXPECT_METRIC_EQ(
        metrics::NumSamples("WebRTC.Audio.EchoCanceller.DelayChanges"),
        expected_num_metric_reports);
    EXPECT_METRIC_EQ(
        metrics::NumSamples("WebRTC.Audio.EchoCanceller.Clockdrift"),
        expected_num_metric_reports);
  }
}

}  // namespace webrtc
