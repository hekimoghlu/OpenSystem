/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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
#ifndef API_TEST_METRICS_METRICS_LOGGER_H_
#define API_TEST_METRICS_METRICS_LOGGER_H_

#include <map>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/numerics/samples_stats_counter.h"
#include "api/test/metrics/metric.h"
#include "api/test/metrics/metrics_accumulator.h"
#include "api/units/timestamp.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {
namespace test {

// Provides API to log and collect performance metrics.
class MetricsLogger {
 public:
  virtual ~MetricsLogger() = default;

  // Adds a metric with a single value.
  // `metadata` - metric's level metadata to add.
  virtual void LogSingleValueMetric(
      absl::string_view name,
      absl::string_view test_case_name,
      double value,
      Unit unit,
      ImprovementDirection improvement_direction,
      std::map<std::string, std::string> metadata = {}) = 0;

  // Adds metrics with a time series created based on the provided `values`.
  // `metadata` - metric's level metadata to add.
  virtual void LogMetric(absl::string_view name,
                         absl::string_view test_case_name,
                         const SamplesStatsCounter& values,
                         Unit unit,
                         ImprovementDirection improvement_direction,
                         std::map<std::string, std::string> metadata = {}) = 0;

  // Adds metric with a time series with only stats object and without actual
  // collected values.
  // `metadata` - metric's level metadata to add.
  virtual void LogMetric(absl::string_view name,
                         absl::string_view test_case_name,
                         const Metric::Stats& metric_stats,
                         Unit unit,
                         ImprovementDirection improvement_direction,
                         std::map<std::string, std::string> metadata = {}) = 0;

  // Returns all metrics collected by this logger.
  virtual std::vector<Metric> GetCollectedMetrics() const = 0;
};

class DefaultMetricsLogger : public MetricsLogger {
 public:
  explicit DefaultMetricsLogger(webrtc::Clock* clock) : clock_(clock) {}
  ~DefaultMetricsLogger() override = default;

  void LogSingleValueMetric(
      absl::string_view name,
      absl::string_view test_case_name,
      double value,
      Unit unit,
      ImprovementDirection improvement_direction,
      std::map<std::string, std::string> metadata = {}) override;

  void LogMetric(absl::string_view name,
                 absl::string_view test_case_name,
                 const SamplesStatsCounter& values,
                 Unit unit,
                 ImprovementDirection improvement_direction,
                 std::map<std::string, std::string> metadata = {}) override;

  void LogMetric(absl::string_view name,
                 absl::string_view test_case_name,
                 const Metric::Stats& metric_stats,
                 Unit unit,
                 ImprovementDirection improvement_direction,
                 std::map<std::string, std::string> metadata = {}) override;

  // Returns all metrics collected by this logger and its `MetricsAccumulator`.
  std::vector<Metric> GetCollectedMetrics() const override;

  MetricsAccumulator* GetMetricsAccumulator() { return &metrics_accumulator_; }

 private:
  webrtc::Timestamp Now();

  webrtc::Clock* const clock_;
  MetricsAccumulator metrics_accumulator_;

  mutable Mutex mutex_;
  std::vector<Metric> metrics_ RTC_GUARDED_BY(mutex_);
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_METRICS_METRICS_LOGGER_H_
