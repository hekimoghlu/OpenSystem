/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
#ifndef API_TEST_METRICS_METRICS_ACCUMULATOR_H_
#define API_TEST_METRICS_METRICS_ACCUMULATOR_H_

#include <map>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/numerics/samples_stats_counter.h"
#include "api/test/metrics/metric.h"
#include "api/units/timestamp.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {
namespace test {

// Accumulates metrics' samples internally and provides API to get collected
// ones.
//
// This object is thread safe.
class MetricsAccumulator {
 public:
  MetricsAccumulator() = default;

  // Adds sample for the specified `metric_name` within specified
  // `test_case_name`. If it is the first time when this combination of
  // `metric_name` and `test_case_name` is used, creates a new Metric to collect
  // samples, otherwise adds a sample to the previously created Metric.
  //
  // By default metric will use `Unit::kUnitless` and
  // `ImprovementDirection::kNeitherIsBetter`.
  //
  // `point_metadata` - the metadata to be added to the single data point that
  // this method adds to the Metric (it is not a metric global metadata).
  //
  // Returns true if a new metric was created and false otherwise.
  bool AddSample(absl::string_view metric_name,
                 absl::string_view test_case_name,
                 double value,
                 Timestamp timestamp,
                 std::map<std::string, std::string> point_metadata = {});

  // Adds metadata to the metric specified by `metric_name` within specified
  // `test_case_name`. If such a metric doesn't exist, creates a new one,
  // otherwise overrides previously recorded values.
  //
  // Returns true if a new metric was created and false otherwise.
  bool AddMetricMetadata(
      absl::string_view metric_name,
      absl::string_view test_case_name,
      Unit unit,
      ImprovementDirection improvement_direction,
      std::map<std::string, std::string> metric_metadata = {});

  // Returns all metrics collected by this accumulator. No order guarantees
  // provided.
  std::vector<Metric> GetCollectedMetrics() const;

 private:
  struct MetricKey {
    MetricKey(absl::string_view metric_name, absl::string_view test_case_name)
        : metric_name(metric_name), test_case_name(test_case_name) {}

    std::string metric_name;
    std::string test_case_name;
  };
  friend bool operator<(const MetricKey& a, const MetricKey& b);

  struct MetricValue {
    SamplesStatsCounter counter;
    Metric metric;
  };

  // Gets existing metrics or creates a new one. If metric was created `created`
  // will be set to true.
  MetricValue* GetOrCreateMetric(absl::string_view metric_name,
                                 absl::string_view test_case_name,
                                 bool* created)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable Mutex mutex_;
  std::map<MetricKey, MetricValue> metrics_ RTC_GUARDED_BY(mutex_);
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_METRICS_METRICS_ACCUMULATOR_H_
