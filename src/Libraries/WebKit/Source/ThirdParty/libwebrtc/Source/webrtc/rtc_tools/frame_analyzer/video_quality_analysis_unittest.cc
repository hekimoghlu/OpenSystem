/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
#include "rtc_tools/frame_analyzer/video_quality_analysis.h"

#include <string>
#include <vector>

#include "api/test/metrics/metric.h"
#include "api/test/metrics/metrics_logger.h"
#include "system_wrappers/include/clock.h"
#include "test/gmock.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace test {
namespace {

using ::testing::IsSupersetOf;

// Metric fields to assert on
struct MetricValidationInfo {
  std::string test_case;
  std::string name;
  Unit unit;
  ImprovementDirection improvement_direction;
  double mean;
};

bool operator==(const MetricValidationInfo& a, const MetricValidationInfo& b) {
  return a.name == b.name && a.test_case == b.test_case && a.unit == b.unit &&
         a.improvement_direction == b.improvement_direction;
}

std::ostream& operator<<(std::ostream& os, const MetricValidationInfo& m) {
  os << "{ test_case=" << m.test_case << "; name=" << m.name
     << "; unit=" << test::ToString(m.unit)
     << "; improvement_direction=" << test::ToString(m.improvement_direction)
     << " }";
  return os;
}

std::vector<MetricValidationInfo> ToValidationInfo(
    const std::vector<Metric>& metrics) {
  std::vector<MetricValidationInfo> out;
  for (const Metric& m : metrics) {
    out.push_back(
        MetricValidationInfo{.test_case = m.test_case,
                             .name = m.name,
                             .unit = m.unit,
                             .improvement_direction = m.improvement_direction,
                             .mean = *m.stats.mean});
  }
  return out;
}

TEST(VideoQualityAnalysisTest, PrintAnalysisResultsEmpty) {
  ResultsContainer result;
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  PrintAnalysisResults("Empty", result, logger);
}

TEST(VideoQualityAnalysisTest, PrintAnalysisResultsOneFrame) {
  ResultsContainer result;
  result.frames.push_back(AnalysisResult(0, 35.0, 0.9));
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  PrintAnalysisResults("OneFrame", result, logger);
}

TEST(VideoQualityAnalysisTest, PrintAnalysisResultsThreeFrames) {
  ResultsContainer result;
  result.frames.push_back(AnalysisResult(0, 35.0, 0.9));
  result.frames.push_back(AnalysisResult(1, 34.0, 0.8));
  result.frames.push_back(AnalysisResult(2, 33.0, 0.7));
  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  PrintAnalysisResults("ThreeFrames", result, logger);
}

TEST(VideoQualityAnalysisTest, PrintMaxRepeatedAndSkippedFramesSkippedFrames) {
  ResultsContainer result;

  result.max_repeated_frames = 2;
  result.max_skipped_frames = 2;
  result.total_skipped_frames = 3;
  result.decode_errors_ref = 0;
  result.decode_errors_test = 0;

  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  PrintAnalysisResults("NormalStatsFile", result, logger);

  std::vector<MetricValidationInfo> metrics =
      ToValidationInfo(logger.GetCollectedMetrics());
  EXPECT_THAT(
      metrics,
      IsSupersetOf(
          {MetricValidationInfo{
               .test_case = "NormalStatsFile",
               .name = "Max_repeated",
               .unit = Unit::kUnitless,
               .improvement_direction = ImprovementDirection::kNeitherIsBetter,
               .mean = 2},
           MetricValidationInfo{
               .test_case = "NormalStatsFile",
               .name = "Max_skipped",
               .unit = Unit::kUnitless,
               .improvement_direction = ImprovementDirection::kNeitherIsBetter,
               .mean = 2},
           MetricValidationInfo{
               .test_case = "NormalStatsFile",
               .name = "Total_skipped",
               .unit = Unit::kUnitless,
               .improvement_direction = ImprovementDirection::kNeitherIsBetter,
               .mean = 3},
           MetricValidationInfo{
               .test_case = "NormalStatsFile",
               .name = "Decode_errors_reference",
               .unit = Unit::kUnitless,
               .improvement_direction = ImprovementDirection::kNeitherIsBetter,
               .mean = 0},
           MetricValidationInfo{
               .test_case = "NormalStatsFile",
               .name = "Decode_errors_test",
               .unit = Unit::kUnitless,
               .improvement_direction = ImprovementDirection::kNeitherIsBetter,
               .mean = 0}}));
}

TEST(VideoQualityAnalysisTest,
     PrintMaxRepeatedAndSkippedFramesDecodeErrorInTest) {
  ResultsContainer result;

  std::string log_filename =
      TempFilename(webrtc::test::OutputPath(), "log.log");
  FILE* logfile = fopen(log_filename.c_str(), "w");
  ASSERT_TRUE(logfile != NULL);

  result.max_repeated_frames = 1;
  result.max_skipped_frames = 0;
  result.total_skipped_frames = 0;
  result.decode_errors_ref = 0;
  result.decode_errors_test = 3;

  DefaultMetricsLogger logger(Clock::GetRealTimeClock());
  PrintAnalysisResults("NormalStatsFile", result, logger);

  std::vector<MetricValidationInfo> metrics =
      ToValidationInfo(logger.GetCollectedMetrics());
  EXPECT_THAT(
      metrics,
      IsSupersetOf(
          {MetricValidationInfo{
               .test_case = "NormalStatsFile",
               .name = "Max_repeated",
               .unit = Unit::kUnitless,
               .improvement_direction = ImprovementDirection::kNeitherIsBetter,
               .mean = 1},
           MetricValidationInfo{
               .test_case = "NormalStatsFile",
               .name = "Max_skipped",
               .unit = Unit::kUnitless,
               .improvement_direction = ImprovementDirection::kNeitherIsBetter,
               .mean = 0},
           MetricValidationInfo{
               .test_case = "NormalStatsFile",
               .name = "Total_skipped",
               .unit = Unit::kUnitless,
               .improvement_direction = ImprovementDirection::kNeitherIsBetter,
               .mean = 0},
           MetricValidationInfo{
               .test_case = "NormalStatsFile",
               .name = "Decode_errors_reference",
               .unit = Unit::kUnitless,
               .improvement_direction = ImprovementDirection::kNeitherIsBetter,
               .mean = 0},
           MetricValidationInfo{
               .test_case = "NormalStatsFile",
               .name = "Decode_errors_test",
               .unit = Unit::kUnitless,
               .improvement_direction = ImprovementDirection::kNeitherIsBetter,
               .mean = 3}}));
}

TEST(VideoQualityAnalysisTest, CalculateFrameClustersOneValue) {
  const std::vector<Cluster> result = CalculateFrameClusters({1});
  EXPECT_EQ(1u, result.size());
  EXPECT_EQ(1u, result[0].index);
  EXPECT_EQ(1, result[0].number_of_repeated_frames);
}

TEST(VideoQualityAnalysisTest, GetMaxRepeatedFramesOneValue) {
  EXPECT_EQ(1, GetMaxRepeatedFrames(CalculateFrameClusters({1})));
}

TEST(VideoQualityAnalysisTest, GetMaxSkippedFramesOneValue) {
  EXPECT_EQ(0, GetMaxSkippedFrames(CalculateFrameClusters({1})));
}

TEST(VideoQualityAnalysisTest, GetTotalNumberOfSkippedFramesOneValue) {
  EXPECT_EQ(0, GetTotalNumberOfSkippedFrames(CalculateFrameClusters({1})));
}

TEST(VideoQualityAnalysisTest, CalculateFrameClustersOneOneTwo) {
  const std::vector<Cluster> result = CalculateFrameClusters({1, 1, 2});
  EXPECT_EQ(2u, result.size());
  EXPECT_EQ(1u, result[0].index);
  EXPECT_EQ(2, result[0].number_of_repeated_frames);
  EXPECT_EQ(2u, result[1].index);
  EXPECT_EQ(1, result[1].number_of_repeated_frames);
}

TEST(VideoQualityAnalysisTest, GetMaxRepeatedFramesOneOneTwo) {
  EXPECT_EQ(2, GetMaxRepeatedFrames(CalculateFrameClusters({1, 1, 2})));
}

TEST(VideoQualityAnalysisTest, GetMaxSkippedFramesOneOneTwo) {
  EXPECT_EQ(0, GetMaxSkippedFrames(CalculateFrameClusters({1, 1, 2})));
}

TEST(VideoQualityAnalysisTest, GetTotalNumberOfSkippedFramesOneOneTwo) {
  EXPECT_EQ(0,
            GetTotalNumberOfSkippedFrames(CalculateFrameClusters({1, 1, 2})));
}

TEST(VideoQualityAnalysisTest, CalculateFrameClustersEmpty) {
  EXPECT_TRUE(CalculateFrameClusters({}).empty());
}

TEST(VideoQualityAnalysisTest, GetMaxRepeatedFramesEmpty) {
  EXPECT_EQ(0, GetMaxRepeatedFrames({}));
}

TEST(VideoQualityAnalysisTest, GetMaxSkippedFramesEmpty) {
  EXPECT_EQ(0, GetMaxSkippedFrames({}));
}

TEST(VideoQualityAnalysisTest, GetTotalNumberOfSkippedFramesEmpty) {
  EXPECT_EQ(0, GetTotalNumberOfSkippedFrames({}));
}

}  // namespace
}  // namespace test
}  // namespace webrtc
