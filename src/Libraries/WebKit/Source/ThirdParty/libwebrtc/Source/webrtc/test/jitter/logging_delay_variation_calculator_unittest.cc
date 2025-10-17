/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
#include "test/jitter/logging_delay_variation_calculator.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "api/test/metrics/metrics_logger.h"
#include "api/units/timestamp.h"
#include "system_wrappers/include/clock.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {

TEST(LoggingDelayVariationCalculator, TimeSeriesWithTwoFrames) {
  SimulatedClock clock(123456);
  DefaultMetricsLogger logger(&clock);
  {
    LoggingDelayVariationCalculator calc("TimeSeriesWithTwoFrames", &logger);
    calc.Insert(3000, Timestamp::Millis(33), DataSize::Bytes(100));
    calc.Insert(6000, Timestamp::Millis(66), DataSize::Bytes(100));
    // Metrics are logged when `calc` goes out of scope.
  }

  std::vector<Metric> metrics = logger.GetCollectedMetrics();
  std::map<std::string, double> last_sample_by_metric_name;
  std::transform(metrics.begin(), metrics.end(),
                 std::inserter(last_sample_by_metric_name,
                               last_sample_by_metric_name.end()),
                 [](const Metric& metric) {
                   EXPECT_FALSE(metric.time_series.samples.empty());
                   return std::make_pair(
                       metric.name, metric.time_series.samples.back().value);
                 });
  EXPECT_EQ(last_sample_by_metric_name["rtp_timestamp"], 6000.0);
  EXPECT_EQ(last_sample_by_metric_name["arrival_time"], 66.0);
  EXPECT_EQ(last_sample_by_metric_name["size_bytes"], 100.0);
  EXPECT_EQ(last_sample_by_metric_name["inter_departure_time"], 33.333);
  EXPECT_EQ(last_sample_by_metric_name["inter_arrival_time"], 33.0);
  EXPECT_EQ(last_sample_by_metric_name["inter_delay_variation"], -0.333);
  EXPECT_EQ(last_sample_by_metric_name["inter_size_variation"], 0.0);
}

}  // namespace test
}  // namespace webrtc
