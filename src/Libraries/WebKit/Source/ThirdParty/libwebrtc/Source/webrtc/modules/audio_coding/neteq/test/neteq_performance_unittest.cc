/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
#include "absl/flags/flag.h"
#include "api/test/metrics/global_metrics_logger_and_exporter.h"
#include "api/test/metrics/metric.h"
#include "modules/audio_coding/neteq/tools/neteq_performance_test.h"
#include "test/gtest.h"
#include "test/test_flags.h"

namespace webrtc {
namespace {

using ::webrtc::test::GetGlobalMetricsLogger;
using ::webrtc::test::ImprovementDirection;
using ::webrtc::test::Unit;

// Runs a test with 10% packet losses and 10% clock drift, to exercise
// both loss concealment and time-stretching code.
TEST(NetEqPerformanceTest, 10_Pl_10_Drift) {
  const int kSimulationTimeMs = 10000000;
  const int kQuickSimulationTimeMs = 100000;
  const int kLossPeriod = 10;  // Drop every 10th packet.
  const double kDriftFactor = 0.1;
  int64_t runtime = test::NetEqPerformanceTest::Run(
      absl::GetFlag(FLAGS_webrtc_quick_perf_test) ? kQuickSimulationTimeMs
                                                  : kSimulationTimeMs,
      kLossPeriod, kDriftFactor);
  ASSERT_GT(runtime, 0);
  GetGlobalMetricsLogger()->LogSingleValueMetric(
      "neteq_performance", "10_pl_10_drift", runtime, Unit::kMilliseconds,
      ImprovementDirection::kNeitherIsBetter);
}

// Runs a test with neither packet losses nor clock drift, to put
// emphasis on the "good-weather" code path, which is presumably much
// more lightweight.
TEST(NetEqPerformanceTest, 0_Pl_0_Drift) {
  const int kSimulationTimeMs = 10000000;
  const int kQuickSimulationTimeMs = 100000;
  const int kLossPeriod = 0;        // No losses.
  const double kDriftFactor = 0.0;  // No clock drift.
  int64_t runtime = test::NetEqPerformanceTest::Run(
      absl::GetFlag(FLAGS_webrtc_quick_perf_test) ? kQuickSimulationTimeMs
                                                  : kSimulationTimeMs,
      kLossPeriod, kDriftFactor);
  ASSERT_GT(runtime, 0);
  GetGlobalMetricsLogger()->LogSingleValueMetric(
      "neteq_performance", "0_pl_0_drift", runtime, Unit::kMilliseconds,
      ImprovementDirection::kNeitherIsBetter);
}

}  // namespace
}  // namespace webrtc
