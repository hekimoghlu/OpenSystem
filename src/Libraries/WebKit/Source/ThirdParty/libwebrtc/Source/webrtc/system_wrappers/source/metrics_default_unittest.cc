/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "rtc_base/checks.h"
#include "rtc_base/string_utils.h"
#include "system_wrappers/include/metrics.h"
#include "test/gtest.h"

#if RTC_METRICS_ENABLED
namespace webrtc {

namespace {
const int kSample = 22;
const char kName[] = "Name";

int NumSamples(absl::string_view name,
               const std::map<std::string,
                              std::unique_ptr<metrics::SampleInfo>,
                              rtc::AbslStringViewCmp>& histograms) {
  const auto it = histograms.find(name);
  if (it == histograms.end())
    return 0;

  int num_samples = 0;
  for (const auto& sample : it->second->samples)
    num_samples += sample.second;

  return num_samples;
}

int NumEvents(absl::string_view name,
              int sample,
              const std::map<std::string,
                             std::unique_ptr<metrics::SampleInfo>,
                             rtc::AbslStringViewCmp>& histograms) {
  const auto it = histograms.find(name);
  if (it == histograms.end())
    return 0;

  const auto it_sample = it->second->samples.find(sample);
  if (it_sample == it->second->samples.end())
    return 0;

  return it_sample->second;
}
}  // namespace

class MetricsDefaultTest : public ::testing::Test {
 public:
  MetricsDefaultTest() {}

 protected:
  void SetUp() override { metrics::Reset(); }
};

TEST_F(MetricsDefaultTest, Reset) {
  RTC_HISTOGRAM_PERCENTAGE(kName, kSample);
  EXPECT_EQ(1, metrics::NumSamples(kName));
  metrics::Reset();
  EXPECT_EQ(0, metrics::NumSamples(kName));
}

TEST_F(MetricsDefaultTest, NumSamples) {
  RTC_HISTOGRAM_PERCENTAGE(kName, 5);
  RTC_HISTOGRAM_PERCENTAGE(kName, 5);
  RTC_HISTOGRAM_PERCENTAGE(kName, 10);
  EXPECT_EQ(3, metrics::NumSamples(kName));
  EXPECT_EQ(0, metrics::NumSamples("NonExisting"));
}

TEST_F(MetricsDefaultTest, NumEvents) {
  RTC_HISTOGRAM_PERCENTAGE(kName, 5);
  RTC_HISTOGRAM_PERCENTAGE(kName, 5);
  RTC_HISTOGRAM_PERCENTAGE(kName, 10);
  EXPECT_EQ(2, metrics::NumEvents(kName, 5));
  EXPECT_EQ(1, metrics::NumEvents(kName, 10));
  EXPECT_EQ(0, metrics::NumEvents(kName, 11));
  EXPECT_EQ(0, metrics::NumEvents("NonExisting", 5));
}

TEST_F(MetricsDefaultTest, MinSample) {
  RTC_HISTOGRAM_PERCENTAGE(kName, kSample);
  RTC_HISTOGRAM_PERCENTAGE(kName, kSample + 1);
  EXPECT_EQ(kSample, metrics::MinSample(kName));
  EXPECT_EQ(-1, metrics::MinSample("NonExisting"));
}

TEST_F(MetricsDefaultTest, Overflow) {
  const std::string kName = "Overflow";
  // Samples should end up in overflow bucket.
  RTC_HISTOGRAM_PERCENTAGE(kName, 101);
  EXPECT_EQ(1, metrics::NumSamples(kName));
  EXPECT_EQ(1, metrics::NumEvents(kName, 101));
  RTC_HISTOGRAM_PERCENTAGE(kName, 102);
  EXPECT_EQ(2, metrics::NumSamples(kName));
  EXPECT_EQ(2, metrics::NumEvents(kName, 101));
}

TEST_F(MetricsDefaultTest, Underflow) {
  const std::string kName = "Underflow";
  // Samples should end up in underflow bucket.
  RTC_HISTOGRAM_COUNTS_10000(kName, 0);
  EXPECT_EQ(1, metrics::NumSamples(kName));
  EXPECT_EQ(1, metrics::NumEvents(kName, 0));
  RTC_HISTOGRAM_COUNTS_10000(kName, -1);
  EXPECT_EQ(2, metrics::NumSamples(kName));
  EXPECT_EQ(2, metrics::NumEvents(kName, 0));
}

TEST_F(MetricsDefaultTest, GetAndReset) {
  std::map<std::string, std::unique_ptr<metrics::SampleInfo>,
           rtc::AbslStringViewCmp>
      histograms;
  metrics::GetAndReset(&histograms);
  EXPECT_EQ(0u, histograms.size());
  RTC_HISTOGRAM_PERCENTAGE("Histogram1", 4);
  RTC_HISTOGRAM_PERCENTAGE("Histogram1", 5);
  RTC_HISTOGRAM_PERCENTAGE("Histogram1", 5);
  RTC_HISTOGRAM_PERCENTAGE("Histogram2", 10);
  EXPECT_EQ(3, metrics::NumSamples("Histogram1"));
  EXPECT_EQ(1, metrics::NumSamples("Histogram2"));

  metrics::GetAndReset(&histograms);
  EXPECT_EQ(2u, histograms.size());
  EXPECT_EQ(0, metrics::NumSamples("Histogram1"));
  EXPECT_EQ(0, metrics::NumSamples("Histogram2"));

  EXPECT_EQ(3, NumSamples("Histogram1", histograms));
  EXPECT_EQ(1, NumSamples("Histogram2", histograms));
  EXPECT_EQ(1, NumEvents("Histogram1", 4, histograms));
  EXPECT_EQ(2, NumEvents("Histogram1", 5, histograms));
  EXPECT_EQ(1, NumEvents("Histogram2", 10, histograms));

  // Add samples after reset.
  metrics::GetAndReset(&histograms);
  EXPECT_EQ(0u, histograms.size());
  RTC_HISTOGRAM_PERCENTAGE("Histogram1", 50);
  RTC_HISTOGRAM_PERCENTAGE("Histogram2", 8);
  EXPECT_EQ(1, metrics::NumSamples("Histogram1"));
  EXPECT_EQ(1, metrics::NumSamples("Histogram2"));
  EXPECT_EQ(1, metrics::NumEvents("Histogram1", 50));
  EXPECT_EQ(1, metrics::NumEvents("Histogram2", 8));
}

TEST_F(MetricsDefaultTest, TestMinMaxBucket) {
  const std::string kName = "MinMaxCounts100";
  RTC_HISTOGRAM_COUNTS_100(kName, 4);

  std::map<std::string, std::unique_ptr<metrics::SampleInfo>,
           rtc::AbslStringViewCmp>
      histograms;
  metrics::GetAndReset(&histograms);
  EXPECT_EQ(1u, histograms.size());
  EXPECT_EQ(kName, histograms.begin()->second->name);
  EXPECT_EQ(1, histograms.begin()->second->min);
  EXPECT_EQ(100, histograms.begin()->second->max);
  EXPECT_EQ(50u, histograms.begin()->second->bucket_count);
  EXPECT_EQ(1u, histograms.begin()->second->samples.size());
}

}  // namespace webrtc
#endif
