/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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
#include "api/test/metrics/metrics_logger.h"

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/numerics/samples_stats_counter.h"
#include "api/test/metrics/metric.h"
#include "api/units/timestamp.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
namespace test {
namespace {

Metric::Stats ToStats(const SamplesStatsCounter& values) {
  if (values.IsEmpty()) {
    return Metric::Stats();
  }
  return Metric::Stats{.mean = values.GetAverage(),
                       .stddev = values.GetStandardDeviation(),
                       .min = values.GetMin(),
                       .max = values.GetMax()};
}

}  // namespace

void DefaultMetricsLogger::LogSingleValueMetric(
    absl::string_view name,
    absl::string_view test_case_name,
    double value,
    Unit unit,
    ImprovementDirection improvement_direction,
    std::map<std::string, std::string> metadata) {
  MutexLock lock(&mutex_);
  metrics_.push_back(Metric{
      .name = std::string(name),
      .unit = unit,
      .improvement_direction = improvement_direction,
      .test_case = std::string(test_case_name),
      .metric_metadata = std::move(metadata),
      .time_series =
          Metric::TimeSeries{.samples = std::vector{Metric::TimeSeries::Sample{
                                 .timestamp = Now(), .value = value}}},
      .stats = Metric::Stats{
          .mean = value, .stddev = std::nullopt, .min = value, .max = value}});
}

void DefaultMetricsLogger::LogMetric(
    absl::string_view name,
    absl::string_view test_case_name,
    const SamplesStatsCounter& values,
    Unit unit,
    ImprovementDirection improvement_direction,
    std::map<std::string, std::string> metadata) {
  MutexLock lock(&mutex_);
  Metric::TimeSeries time_series;
  for (const SamplesStatsCounter::StatsSample& sample :
       values.GetTimedSamples()) {
    time_series.samples.push_back(
        Metric::TimeSeries::Sample{.timestamp = sample.time,
                                   .value = sample.value,
                                   .sample_metadata = sample.metadata});
  }

  metrics_.push_back(Metric{.name = std::string(name),
                            .unit = unit,
                            .improvement_direction = improvement_direction,
                            .test_case = std::string(test_case_name),
                            .metric_metadata = std::move(metadata),
                            .time_series = std::move(time_series),
                            .stats = ToStats(values)});
}

void DefaultMetricsLogger::LogMetric(
    absl::string_view name,
    absl::string_view test_case_name,
    const Metric::Stats& metric_stats,
    Unit unit,
    ImprovementDirection improvement_direction,
    std::map<std::string, std::string> metadata) {
  MutexLock lock(&mutex_);
  metrics_.push_back(Metric{.name = std::string(name),
                            .unit = unit,
                            .improvement_direction = improvement_direction,
                            .test_case = std::string(test_case_name),
                            .metric_metadata = std::move(metadata),
                            .time_series = Metric::TimeSeries{.samples = {}},
                            .stats = std::move(metric_stats)});
}

std::vector<Metric> DefaultMetricsLogger::GetCollectedMetrics() const {
  std::vector<Metric> out = metrics_accumulator_.GetCollectedMetrics();
  MutexLock lock(&mutex_);
  out.insert(out.end(), metrics_.begin(), metrics_.end());
  return out;
}

Timestamp DefaultMetricsLogger::Now() {
  return clock_->CurrentTime();
}

}  // namespace test
}  // namespace webrtc
