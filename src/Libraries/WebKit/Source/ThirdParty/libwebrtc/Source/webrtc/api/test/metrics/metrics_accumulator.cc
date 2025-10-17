/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 29, 2022.
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
#include "api/test/metrics/metrics_accumulator.h"

#include <map>
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

Metric SetTimeseries(const Metric& prototype,
                     const SamplesStatsCounter& counter) {
  Metric output(prototype);
  Metric::TimeSeries time_series;
  for (const SamplesStatsCounter::StatsSample& sample :
       counter.GetTimedSamples()) {
    time_series.samples.push_back(
        Metric::TimeSeries::Sample{.timestamp = sample.time,
                                   .value = sample.value,
                                   .sample_metadata = sample.metadata});
  }
  output.time_series = std::move(time_series);
  output.stats = ToStats(counter);
  return output;
}

}  // namespace

bool operator<(const MetricsAccumulator::MetricKey& a,
               const MetricsAccumulator::MetricKey& b) {
  if (a.test_case_name < b.test_case_name) {
    return true;
  } else if (a.test_case_name > b.test_case_name) {
    return false;
  } else {
    return a.metric_name < b.metric_name;
  }
}

bool MetricsAccumulator::AddSample(
    absl::string_view metric_name,
    absl::string_view test_case_name,
    double value,
    Timestamp timestamp,
    std::map<std::string, std::string> point_metadata) {
  MutexLock lock(&mutex_);
  bool created;
  MetricValue* metric_value =
      GetOrCreateMetric(metric_name, test_case_name, &created);
  metric_value->counter.AddSample(
      SamplesStatsCounter::StatsSample{.value = value,
                                       .time = timestamp,
                                       .metadata = std::move(point_metadata)});
  return created;
}

bool MetricsAccumulator::AddMetricMetadata(
    absl::string_view metric_name,
    absl::string_view test_case_name,
    Unit unit,
    ImprovementDirection improvement_direction,
    std::map<std::string, std::string> metric_metadata) {
  MutexLock lock(&mutex_);
  bool created;
  MetricValue* metric_value =
      GetOrCreateMetric(metric_name, test_case_name, &created);
  metric_value->metric.unit = unit;
  metric_value->metric.improvement_direction = improvement_direction;
  metric_value->metric.metric_metadata = std::move(metric_metadata);
  return created;
}

std::vector<Metric> MetricsAccumulator::GetCollectedMetrics() const {
  MutexLock lock(&mutex_);
  std::vector<Metric> out;
  out.reserve(metrics_.size());
  for (const auto& [unused_key, metric_value] : metrics_) {
    out.push_back(SetTimeseries(metric_value.metric, metric_value.counter));
  }
  return out;
}

MetricsAccumulator::MetricValue* MetricsAccumulator::GetOrCreateMetric(
    absl::string_view metric_name,
    absl::string_view test_case_name,
    bool* created) {
  MetricKey key(metric_name, test_case_name);
  auto it = metrics_.find(key);
  if (it != metrics_.end()) {
    *created = false;
    return &it->second;
  }
  *created = true;

  Metric metric{
      .name = key.metric_name,
      .unit = Unit::kUnitless,
      .improvement_direction = ImprovementDirection::kNeitherIsBetter,
      .test_case = key.test_case_name,
  };
  return &metrics_.emplace(key, MetricValue{.metric = std::move(metric)})
              .first->second;
}

}  // namespace test
}  // namespace webrtc
