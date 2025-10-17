/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#include "test/testsupport/perf_test.h"

#include <algorithm>
#include <limits>
#include <string>

#include "test/gmock.h"
#include "test/gtest.h"
#include "test/testsupport/rtc_expect_death.h"

#if WEBRTC_ENABLE_PROTOBUF
#include "third_party/catapult/tracing/tracing/value/histogram.h"
namespace proto = catapult::tracing::tracing::proto;
#endif

namespace webrtc {
namespace test {

class PerfTest : public ::testing::Test {
 protected:
  void TearDown() override { ClearPerfResults(); }
};

#if defined(WEBRTC_IOS)
#define MAYBE_TestPrintResult DISABLED_TestPrintResult
#else
#define MAYBE_TestPrintResult TestPrintResult
#endif
TEST_F(PerfTest, MAYBE_TestPrintResult) {
  ::testing::internal::CaptureStdout();
  std::string expected;

  expected += "RESULT measurementmodifier: trace= 42 units\n";
  PrintResult("measurement", "modifier", "trace", 42, "units", false);

  expected += "*RESULT foobar: baz_v= 1423730 widgets\n";
  PrintResult("foo", "bar", "baz_v", 1423730, "widgets", true);

  expected += "RESULT foobar: baz_me= {1,2} lemurs\n";
  PrintResultMeanAndError("foo", "bar", "baz_me", 1, 2, "lemurs", false);

  const double kListOfScalars[] = {1, 2, 3};
  expected += "RESULT foobar: baz_vl= [1,2,3] units\n";
  PrintResultList("foo", "bar", "baz_vl", kListOfScalars, "units", false);

  EXPECT_EQ(expected, ::testing::internal::GetCapturedStdout());
}

TEST_F(PerfTest, TestClearPerfResults) {
  PrintResult("measurement", "modifier", "trace", 42, "units", false);
  ClearPerfResults();
  EXPECT_EQ("", GetPerfResults());
}

#if WEBRTC_ENABLE_PROTOBUF

TEST_F(PerfTest, TestGetPerfResultsHistograms) {
  ClearPerfResults();
  PrintResult("measurement", "_modifier", "story_1", 42, "ms", false);
  PrintResult("foo", "bar", "story_1", 7, "sigma", true);
  // Note: the error will be ignored, not supported by histograms.
  PrintResultMeanAndError("foo", "bar", "story_1", 1, 2000, "sigma", false);
  const double kListOfScalars[] = {1, 2, 3};
  PrintResultList("foo", "bar", "story_1", kListOfScalars, "sigma", false);

  proto::HistogramSet histogram_set;
  EXPECT_TRUE(histogram_set.ParseFromString(GetPerfResults()))
      << "Expected valid histogram set";

  ASSERT_EQ(histogram_set.histograms_size(), 2)
      << "Should be two histograms: foobar and measurement_modifier";
  const proto::Histogram& hist1 = histogram_set.histograms(0);
  const proto::Histogram& hist2 = histogram_set.histograms(1);

  EXPECT_EQ(hist1.name(), "foobar");

  // Spot check some things in here (there's a more thorough test on the
  // histogram writer itself).
  EXPECT_EQ(hist1.unit().unit(), proto::SIGMA);
  EXPECT_EQ(hist1.sample_values_size(), 5);
  EXPECT_EQ(hist1.sample_values(0), 7);
  EXPECT_EQ(hist1.sample_values(1), 1);
  EXPECT_EQ(hist1.sample_values(2), 1);
  EXPECT_EQ(hist1.sample_values(3), 2);
  EXPECT_EQ(hist1.sample_values(4), 3);

  EXPECT_EQ(hist1.diagnostics().diagnostic_map().count("stories"), 1u);
  const proto::Diagnostic& stories =
      hist1.diagnostics().diagnostic_map().at("stories");
  ASSERT_EQ(stories.generic_set().values_size(), 1);
  EXPECT_EQ(stories.generic_set().values(0), "\"story_1\"");

  EXPECT_EQ(hist2.name(), "measurement_modifier");
  EXPECT_EQ(hist2.unit().unit(), proto::MS_BEST_FIT_FORMAT);
}

TEST_F(PerfTest, TestGetPerfResultsHistogramsWithEmptyCounter) {
  ClearPerfResults();
  ::testing::internal::CaptureStdout();

  SamplesStatsCounter empty_counter;
  PrintResult("measurement", "_modifier", "story", empty_counter, "ms", false);

  proto::HistogramSet histogram_set;
  EXPECT_TRUE(histogram_set.ParseFromString(GetPerfResults()))
      << "Expected valid histogram set";

  ASSERT_EQ(histogram_set.histograms_size(), 1)
      << "Should be one histogram: measurement_modifier";
  const proto::Histogram& hist = histogram_set.histograms(0);

  EXPECT_EQ(hist.name(), "measurement_modifier");

  // Spot check some things in here (there's a more thorough test on the
  // histogram writer itself).
  EXPECT_EQ(hist.unit().unit(), proto::MS_BEST_FIT_FORMAT);
  EXPECT_EQ(hist.sample_values_size(), 1);
  EXPECT_EQ(hist.sample_values(0), 0);

  EXPECT_EQ(hist.diagnostics().diagnostic_map().count("stories"), 1u);
  const proto::Diagnostic& stories =
      hist.diagnostics().diagnostic_map().at("stories");
  ASSERT_EQ(stories.generic_set().values_size(), 1);
  EXPECT_EQ(stories.generic_set().values(0), "\"story\"");

  std::string expected = "RESULT measurement_modifier: story= {0,0} ms\n";
  EXPECT_EQ(expected, ::testing::internal::GetCapturedStdout());
}

TEST_F(PerfTest, TestGetPerfResultsHistogramsWithStatsCounter) {
  ClearPerfResults();
  ::testing::internal::CaptureStdout();

  SamplesStatsCounter counter;
  counter.AddSample(1);
  counter.AddSample(2);
  counter.AddSample(3);
  counter.AddSample(4);
  counter.AddSample(5);
  PrintResult("measurement", "_modifier", "story", counter, "ms", false);

  proto::HistogramSet histogram_set;
  EXPECT_TRUE(histogram_set.ParseFromString(GetPerfResults()))
      << "Expected valid histogram set";

  ASSERT_EQ(histogram_set.histograms_size(), 1)
      << "Should be one histogram: measurement_modifier";
  const proto::Histogram& hist = histogram_set.histograms(0);

  EXPECT_EQ(hist.name(), "measurement_modifier");

  // Spot check some things in here (there's a more thorough test on the
  // histogram writer itself).
  EXPECT_EQ(hist.unit().unit(), proto::MS_BEST_FIT_FORMAT);
  EXPECT_EQ(hist.sample_values_size(), 5);
  EXPECT_THAT(hist.sample_values(), testing::ElementsAre(1, 2, 3, 4, 5));

  EXPECT_EQ(hist.diagnostics().diagnostic_map().count("stories"), 1u);
  const proto::Diagnostic& stories =
      hist.diagnostics().diagnostic_map().at("stories");
  ASSERT_EQ(stories.generic_set().values_size(), 1);
  EXPECT_EQ(stories.generic_set().values(0), "\"story\"");

  // mean = 3; std = sqrt(2)
  std::string expected =
      "RESULT measurement_modifier: story= {3,1.4142136} ms\n";
  EXPECT_EQ(expected, ::testing::internal::GetCapturedStdout());
}

#endif  // WEBRTC_ENABLE_PROTOBUF

#if GTEST_HAS_DEATH_TEST
using PerfDeathTest = PerfTest;

TEST_F(PerfDeathTest, TestFiniteResultError) {
  const double kNan = std::numeric_limits<double>::quiet_NaN();
  const double kInf = std::numeric_limits<double>::infinity();

  RTC_EXPECT_DEATH(PrintResult("a", "b", "c", kNan, "d", false), "finit");
  RTC_EXPECT_DEATH(PrintResult("a", "b", "c", kInf, "d", false), "finit");

  RTC_EXPECT_DEATH(PrintResultMeanAndError("a", "b", "c", kNan, 1, "d", false),
                   "");
  RTC_EXPECT_DEATH(PrintResultMeanAndError("a", "b", "c", 1, kInf, "d", false),
                   "");

  const double kNanList[] = {kNan, kNan};
  RTC_EXPECT_DEATH(PrintResultList("a", "b", "c", kNanList, "d", false), "");
  const double kInfList[] = {0, kInf};
  RTC_EXPECT_DEATH(PrintResultList("a", "b", "c", kInfList, "d", false), "");
}
#endif

}  // namespace test
}  // namespace webrtc
