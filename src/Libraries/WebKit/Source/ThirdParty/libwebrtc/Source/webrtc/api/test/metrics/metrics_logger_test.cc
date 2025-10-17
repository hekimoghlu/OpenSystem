/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#include <vector>

#include "api/numerics/samples_stats_counter.h"
#include "api/test/metrics/metric.h"
#include "api/units/timestamp.h"
#include "system_wrappers/include/clock.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {
namespace {

using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::SizeIs;

std::map<std::string, std::string> DefaultMetadata() {
  return std::map<std::string, std::string>{{"key", "value"}};
}

TEST(DefaultMetricsLoggerTest, LogSingleValueMetricRecordsMetric) {
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  logger.LogSingleValueMetric(
      "metric_name", "test_case_name",
      /*value=*/10, Unit::kMilliseconds, ImprovementDirection::kBiggerIsBetter,
      std::map<std::string, std::string>{{"key", "value"}});

  std::vector<Metric> metrics = logger.GetCollectedMetrics();
  ASSERT_THAT(metrics, SizeIs(1));
  const Metric& metric = metrics[0];
  EXPECT_THAT(metric.name, Eq("metric_name"));
  EXPECT_THAT(metric.test_case, Eq("test_case_name"));
  EXPECT_THAT(metric.unit, Eq(Unit::kMilliseconds));
  EXPECT_THAT(metric.improvement_direction,
              Eq(ImprovementDirection::kBiggerIsBetter));
  EXPECT_THAT(metric.metric_metadata,
              Eq(std::map<std::string, std::string>{{"key", "value"}}));
  ASSERT_THAT(metric.time_series.samples, SizeIs(1));
  EXPECT_THAT(metric.time_series.samples[0].value, Eq(10.0));
  EXPECT_THAT(metric.time_series.samples[0].sample_metadata,
              Eq(std::map<std::string, std::string>{}));
  ASSERT_THAT(metric.stats.mean, std::optional<double>(10.0));
  ASSERT_THAT(metric.stats.stddev, std::nullopt);
  ASSERT_THAT(metric.stats.min, std::optional<double>(10.0));
  ASSERT_THAT(metric.stats.max, std::optional<double>(10.0));
}

TEST(DefaultMetricsLoggerTest, LogMetricWithSamplesStatsCounterRecordsMetric) {
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());

  SamplesStatsCounter values;
  values.AddSample(SamplesStatsCounter::StatsSample{
      .value = 10,
      .time = Clock::GetRealTimeClock()->CurrentTime(),
      .metadata =
          std::map<std::string, std::string>{{"point_key1", "value1"}}});
  values.AddSample(SamplesStatsCounter::StatsSample{
      .value = 20,
      .time = Clock::GetRealTimeClock()->CurrentTime(),
      .metadata =
          std::map<std::string, std::string>{{"point_key2", "value2"}}});
  logger.LogMetric("metric_name", "test_case_name", values, Unit::kMilliseconds,
                   ImprovementDirection::kBiggerIsBetter,
                   std::map<std::string, std::string>{{"key", "value"}});

  std::vector<Metric> metrics = logger.GetCollectedMetrics();
  ASSERT_THAT(metrics, SizeIs(1));
  const Metric& metric = metrics[0];
  EXPECT_THAT(metric.name, Eq("metric_name"));
  EXPECT_THAT(metric.test_case, Eq("test_case_name"));
  EXPECT_THAT(metric.unit, Eq(Unit::kMilliseconds));
  EXPECT_THAT(metric.improvement_direction,
              Eq(ImprovementDirection::kBiggerIsBetter));
  EXPECT_THAT(metric.metric_metadata,
              Eq(std::map<std::string, std::string>{{"key", "value"}}));
  ASSERT_THAT(metric.time_series.samples, SizeIs(2));
  EXPECT_THAT(metric.time_series.samples[0].value, Eq(10.0));
  EXPECT_THAT(metric.time_series.samples[0].sample_metadata,
              Eq(std::map<std::string, std::string>{{"point_key1", "value1"}}));
  EXPECT_THAT(metric.time_series.samples[1].value, Eq(20.0));
  EXPECT_THAT(metric.time_series.samples[1].sample_metadata,
              Eq(std::map<std::string, std::string>{{"point_key2", "value2"}}));
  ASSERT_THAT(metric.stats.mean, std::optional<double>(15.0));
  ASSERT_THAT(metric.stats.stddev, std::optional<double>(5.0));
  ASSERT_THAT(metric.stats.min, std::optional<double>(10.0));
  ASSERT_THAT(metric.stats.max, std::optional<double>(20.0));
}

TEST(DefaultMetricsLoggerTest,
     LogMetricWithEmptySamplesStatsCounterRecordsEmptyMetric) {
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  SamplesStatsCounter values;
  logger.LogMetric("metric_name", "test_case_name", values, Unit::kUnitless,
                   ImprovementDirection::kBiggerIsBetter, DefaultMetadata());

  std::vector<Metric> metrics = logger.GetCollectedMetrics();
  ASSERT_THAT(metrics, SizeIs(1));
  EXPECT_THAT(metrics[0].name, Eq("metric_name"));
  EXPECT_THAT(metrics[0].test_case, Eq("test_case_name"));
  EXPECT_THAT(metrics[0].time_series.samples, IsEmpty());
  ASSERT_THAT(metrics[0].stats.mean, Eq(std::nullopt));
  ASSERT_THAT(metrics[0].stats.stddev, Eq(std::nullopt));
  ASSERT_THAT(metrics[0].stats.min, Eq(std::nullopt));
  ASSERT_THAT(metrics[0].stats.max, Eq(std::nullopt));
}

TEST(DefaultMetricsLoggerTest, LogMetricWithStatsRecordsMetric) {
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  Metric::Stats metric_stats{.mean = 15, .stddev = 5, .min = 10, .max = 20};
  logger.LogMetric("metric_name", "test_case_name", metric_stats,
                   Unit::kMilliseconds, ImprovementDirection::kBiggerIsBetter,
                   std::map<std::string, std::string>{{"key", "value"}});

  std::vector<Metric> metrics = logger.GetCollectedMetrics();
  ASSERT_THAT(metrics, SizeIs(1));
  const Metric& metric = metrics[0];
  EXPECT_THAT(metric.name, Eq("metric_name"));
  EXPECT_THAT(metric.test_case, Eq("test_case_name"));
  EXPECT_THAT(metric.unit, Eq(Unit::kMilliseconds));
  EXPECT_THAT(metric.improvement_direction,
              Eq(ImprovementDirection::kBiggerIsBetter));
  EXPECT_THAT(metric.metric_metadata,
              Eq(std::map<std::string, std::string>{{"key", "value"}}));
  ASSERT_THAT(metric.time_series.samples, IsEmpty());
  ASSERT_THAT(metric.stats.mean, std::optional<double>(15.0));
  ASSERT_THAT(metric.stats.stddev, std::optional<double>(5.0));
  ASSERT_THAT(metric.stats.min, std::optional<double>(10.0));
  ASSERT_THAT(metric.stats.max, std::optional<double>(20.0));
}

TEST(DefaultMetricsLoggerTest, LogSingleValueMetricRecordsMultipleMetrics) {
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());

  logger.LogSingleValueMetric("metric_name1", "test_case_name1",
                              /*value=*/10, Unit::kMilliseconds,
                              ImprovementDirection::kBiggerIsBetter,
                              DefaultMetadata());
  logger.LogSingleValueMetric("metric_name2", "test_case_name2",
                              /*value=*/10, Unit::kMilliseconds,
                              ImprovementDirection::kBiggerIsBetter,
                              DefaultMetadata());

  std::vector<Metric> metrics = logger.GetCollectedMetrics();
  ASSERT_THAT(metrics, SizeIs(2));
  EXPECT_THAT(metrics[0].name, Eq("metric_name1"));
  EXPECT_THAT(metrics[0].test_case, Eq("test_case_name1"));
  EXPECT_THAT(metrics[1].name, Eq("metric_name2"));
  EXPECT_THAT(metrics[1].test_case, Eq("test_case_name2"));
}

TEST(DefaultMetricsLoggerTest,
     LogMetricWithSamplesStatsCounterRecordsMultipleMetrics) {
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  SamplesStatsCounter values;
  values.AddSample(SamplesStatsCounter::StatsSample{
      .value = 10,
      .time = Clock::GetRealTimeClock()->CurrentTime(),
      .metadata = DefaultMetadata()});
  values.AddSample(SamplesStatsCounter::StatsSample{
      .value = 20,
      .time = Clock::GetRealTimeClock()->CurrentTime(),
      .metadata = DefaultMetadata()});

  logger.LogMetric("metric_name1", "test_case_name1", values,
                   Unit::kMilliseconds, ImprovementDirection::kBiggerIsBetter,
                   DefaultMetadata());
  logger.LogMetric("metric_name2", "test_case_name2", values,
                   Unit::kMilliseconds, ImprovementDirection::kBiggerIsBetter,
                   DefaultMetadata());

  std::vector<Metric> metrics = logger.GetCollectedMetrics();
  ASSERT_THAT(metrics, SizeIs(2));
  EXPECT_THAT(metrics[0].name, Eq("metric_name1"));
  EXPECT_THAT(metrics[0].test_case, Eq("test_case_name1"));
  EXPECT_THAT(metrics[1].name, Eq("metric_name2"));
  EXPECT_THAT(metrics[1].test_case, Eq("test_case_name2"));
}

TEST(DefaultMetricsLoggerTest, LogMetricWithStatsRecordsMultipleMetrics) {
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  Metric::Stats metric_stats{.mean = 15, .stddev = 5, .min = 10, .max = 20};

  logger.LogMetric("metric_name1", "test_case_name1", metric_stats,
                   Unit::kMilliseconds, ImprovementDirection::kBiggerIsBetter,
                   DefaultMetadata());
  logger.LogMetric("metric_name2", "test_case_name2", metric_stats,
                   Unit::kMilliseconds, ImprovementDirection::kBiggerIsBetter,
                   DefaultMetadata());

  std::vector<Metric> metrics = logger.GetCollectedMetrics();
  ASSERT_THAT(metrics, SizeIs(2));
  EXPECT_THAT(metrics[0].name, Eq("metric_name1"));
  EXPECT_THAT(metrics[0].test_case, Eq("test_case_name1"));
  EXPECT_THAT(metrics[1].name, Eq("metric_name2"));
  EXPECT_THAT(metrics[1].test_case, Eq("test_case_name2"));
}

TEST(DefaultMetricsLoggerTest,
     LogMetricThroughtAllMethodsAccumulateAllMetrics) {
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  SamplesStatsCounter values;
  values.AddSample(SamplesStatsCounter::StatsSample{
      .value = 10,
      .time = Clock::GetRealTimeClock()->CurrentTime(),
      .metadata = DefaultMetadata()});
  values.AddSample(SamplesStatsCounter::StatsSample{
      .value = 20,
      .time = Clock::GetRealTimeClock()->CurrentTime(),
      .metadata = DefaultMetadata()});
  Metric::Stats metric_stats{.mean = 15, .stddev = 5, .min = 10, .max = 20};

  logger.LogSingleValueMetric("metric_name1", "test_case_name1",
                              /*value=*/10, Unit::kMilliseconds,
                              ImprovementDirection::kBiggerIsBetter,
                              DefaultMetadata());
  logger.LogMetric("metric_name2", "test_case_name2", values,
                   Unit::kMilliseconds, ImprovementDirection::kBiggerIsBetter,
                   DefaultMetadata());
  logger.LogMetric("metric_name3", "test_case_name3", metric_stats,
                   Unit::kMilliseconds, ImprovementDirection::kBiggerIsBetter,
                   DefaultMetadata());

  std::vector<Metric> metrics = logger.GetCollectedMetrics();
  ASSERT_THAT(metrics.size(), Eq(3lu));
  EXPECT_THAT(metrics[0].name, Eq("metric_name1"));
  EXPECT_THAT(metrics[0].test_case, Eq("test_case_name1"));
  EXPECT_THAT(metrics[1].name, Eq("metric_name2"));
  EXPECT_THAT(metrics[1].test_case, Eq("test_case_name2"));
  EXPECT_THAT(metrics[2].name, Eq("metric_name3"));
  EXPECT_THAT(metrics[2].test_case, Eq("test_case_name3"));
}

TEST(DefaultMetricsLoggerTest, AccumulatedMetricsReturnedInCollectedMetrics) {
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  logger.GetMetricsAccumulator()->AddSample(
      "metric_name", "test_case_name",
      /*value=*/10, Timestamp::Seconds(1),
      /*point_metadata=*/std::map<std::string, std::string>{{"key", "value"}});

  std::vector<Metric> metrics = logger.GetCollectedMetrics();
  ASSERT_THAT(metrics, SizeIs(1));
  const Metric& metric = metrics[0];
  EXPECT_THAT(metric.name, Eq("metric_name"));
  EXPECT_THAT(metric.test_case, Eq("test_case_name"));
  EXPECT_THAT(metric.unit, Eq(Unit::kUnitless));
  EXPECT_THAT(metric.improvement_direction,
              Eq(ImprovementDirection::kNeitherIsBetter));
  EXPECT_THAT(metric.metric_metadata, IsEmpty());
  ASSERT_THAT(metric.time_series.samples, SizeIs(1));
  EXPECT_THAT(metric.time_series.samples[0].value, Eq(10.0));
  EXPECT_THAT(metric.time_series.samples[0].timestamp,
              Eq(Timestamp::Seconds(1)));
  EXPECT_THAT(metric.time_series.samples[0].sample_metadata,
              Eq(std::map<std::string, std::string>{{"key", "value"}}));
  ASSERT_THAT(metric.stats.mean, std::optional<double>(10.0));
  ASSERT_THAT(metric.stats.stddev, std::optional<double>(0.0));
  ASSERT_THAT(metric.stats.min, std::optional<double>(10.0));
  ASSERT_THAT(metric.stats.max, std::optional<double>(10.0));
}

TEST(DefaultMetricsLoggerTest,
     AccumulatedMetricsReturnedTogetherWithLoggedMetrics) {
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  logger.LogSingleValueMetric(
      "metric_name1", "test_case_name1",
      /*value=*/10, Unit::kMilliseconds, ImprovementDirection::kBiggerIsBetter,
      std::map<std::string, std::string>{{"key_m", "value_m"}});
  logger.GetMetricsAccumulator()->AddSample(
      "metric_name2", "test_case_name2",
      /*value=*/10, Timestamp::Seconds(1),
      /*point_metadata=*/
      std::map<std::string, std::string>{{"key_s", "value_s"}});

  std::vector<Metric> metrics = logger.GetCollectedMetrics();
  ASSERT_THAT(metrics, SizeIs(2));
  EXPECT_THAT(metrics[0].name, Eq("metric_name2"));
  EXPECT_THAT(metrics[0].test_case, Eq("test_case_name2"));
  EXPECT_THAT(metrics[0].unit, Eq(Unit::kUnitless));
  EXPECT_THAT(metrics[0].improvement_direction,
              Eq(ImprovementDirection::kNeitherIsBetter));
  EXPECT_THAT(metrics[0].metric_metadata, IsEmpty());
  ASSERT_THAT(metrics[0].time_series.samples, SizeIs(1));
  EXPECT_THAT(metrics[0].time_series.samples[0].value, Eq(10.0));
  EXPECT_THAT(metrics[0].time_series.samples[0].timestamp,
              Eq(Timestamp::Seconds(1)));
  EXPECT_THAT(metrics[0].time_series.samples[0].sample_metadata,
              Eq(std::map<std::string, std::string>{{"key_s", "value_s"}}));
  ASSERT_THAT(metrics[0].stats.mean, std::optional<double>(10.0));
  ASSERT_THAT(metrics[0].stats.stddev, std::optional<double>(0.0));
  ASSERT_THAT(metrics[0].stats.min, std::optional<double>(10.0));
  ASSERT_THAT(metrics[0].stats.max, std::optional<double>(10.0));
  EXPECT_THAT(metrics[1].name, Eq("metric_name1"));
  EXPECT_THAT(metrics[1].test_case, Eq("test_case_name1"));
  EXPECT_THAT(metrics[1].unit, Eq(Unit::kMilliseconds));
  EXPECT_THAT(metrics[1].improvement_direction,
              Eq(ImprovementDirection::kBiggerIsBetter));
  EXPECT_THAT(metrics[1].metric_metadata,
              Eq(std::map<std::string, std::string>{{"key_m", "value_m"}}));
  ASSERT_THAT(metrics[1].time_series.samples, SizeIs(1));
  EXPECT_THAT(metrics[1].time_series.samples[0].value, Eq(10.0));
  EXPECT_THAT(metrics[1].time_series.samples[0].sample_metadata,
              Eq(std::map<std::string, std::string>{}));
  ASSERT_THAT(metrics[1].stats.mean, std::optional<double>(10.0));
  ASSERT_THAT(metrics[1].stats.stddev, std::nullopt);
  ASSERT_THAT(metrics[1].stats.min, std::optional<double>(10.0));
  ASSERT_THAT(metrics[1].stats.max, std::optional<double>(10.0));
}

}  // namespace
}  // namespace test
}  // namespace webrtc
