/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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
#include "modules/congestion_controller/goog_cc/alr_detector.h"

#include <cstdint>
#include <optional>

#include "api/transport/field_trial_based_config.h"
#include "rtc_base/checks.h"
#include "rtc_base/experiments/alr_experiment.h"
#include "test/field_trial.h"
#include "test/gtest.h"

namespace {

constexpr int kEstimatedBitrateBps = 300000;

}  // namespace

namespace webrtc {
namespace {
class SimulateOutgoingTrafficIn {
 public:
  explicit SimulateOutgoingTrafficIn(AlrDetector* alr_detector,
                                     int64_t* timestamp_ms)
      : alr_detector_(alr_detector), timestamp_ms_(timestamp_ms) {
    RTC_CHECK(alr_detector_);
  }

  SimulateOutgoingTrafficIn& ForTimeMs(int time_ms) {
    interval_ms_ = time_ms;
    ProduceTraffic();
    return *this;
  }

  SimulateOutgoingTrafficIn& AtPercentOfEstimatedBitrate(int usage_percentage) {
    usage_percentage_.emplace(usage_percentage);
    ProduceTraffic();
    return *this;
  }

 private:
  void ProduceTraffic() {
    if (!interval_ms_ || !usage_percentage_)
      return;
    const int kTimeStepMs = 10;
    for (int t = 0; t < *interval_ms_; t += kTimeStepMs) {
      *timestamp_ms_ += kTimeStepMs;
      alr_detector_->OnBytesSent(kEstimatedBitrateBps * *usage_percentage_ *
                                     kTimeStepMs / (8 * 100 * 1000),
                                 *timestamp_ms_);
    }
    int remainder_ms = *interval_ms_ % kTimeStepMs;
    if (remainder_ms > 0) {
      *timestamp_ms_ += kTimeStepMs;
      alr_detector_->OnBytesSent(kEstimatedBitrateBps * *usage_percentage_ *
                                     remainder_ms / (8 * 100 * 1000),
                                 *timestamp_ms_);
    }
  }
  AlrDetector* const alr_detector_;
  int64_t* timestamp_ms_;
  std::optional<int> interval_ms_;
  std::optional<int> usage_percentage_;
};
}  // namespace

TEST(AlrDetectorTest, AlrDetection) {
  FieldTrialBasedConfig field_trials;
  int64_t timestamp_ms = 1000;
  AlrDetector alr_detector(&field_trials);
  alr_detector.SetEstimatedBitrate(kEstimatedBitrateBps);

  // Start in non-ALR state.
  EXPECT_FALSE(alr_detector.GetApplicationLimitedRegionStartTime());

  // Stay in non-ALR state when usage is close to 100%.
  SimulateOutgoingTrafficIn(&alr_detector, &timestamp_ms)
      .ForTimeMs(1000)
      .AtPercentOfEstimatedBitrate(90);
  EXPECT_FALSE(alr_detector.GetApplicationLimitedRegionStartTime());

  // Verify that we ALR starts when bitrate drops below 20%.
  SimulateOutgoingTrafficIn(&alr_detector, &timestamp_ms)
      .ForTimeMs(1500)
      .AtPercentOfEstimatedBitrate(20);
  EXPECT_TRUE(alr_detector.GetApplicationLimitedRegionStartTime());

  // Verify that ALR ends when usage is above 65%.
  SimulateOutgoingTrafficIn(&alr_detector, &timestamp_ms)
      .ForTimeMs(4000)
      .AtPercentOfEstimatedBitrate(100);
  EXPECT_FALSE(alr_detector.GetApplicationLimitedRegionStartTime());
}

TEST(AlrDetectorTest, ShortSpike) {
  FieldTrialBasedConfig field_trials;
  int64_t timestamp_ms = 1000;
  AlrDetector alr_detector(&field_trials);
  alr_detector.SetEstimatedBitrate(kEstimatedBitrateBps);
  // Start in non-ALR state.
  EXPECT_FALSE(alr_detector.GetApplicationLimitedRegionStartTime());

  // Verify that we ALR starts when bitrate drops below 20%.
  SimulateOutgoingTrafficIn(&alr_detector, &timestamp_ms)
      .ForTimeMs(1000)
      .AtPercentOfEstimatedBitrate(20);
  EXPECT_TRUE(alr_detector.GetApplicationLimitedRegionStartTime());

  // Verify that we stay in ALR region even after a short bitrate spike.
  SimulateOutgoingTrafficIn(&alr_detector, &timestamp_ms)
      .ForTimeMs(100)
      .AtPercentOfEstimatedBitrate(150);
  EXPECT_TRUE(alr_detector.GetApplicationLimitedRegionStartTime());

  // ALR ends when usage is above 65%.
  SimulateOutgoingTrafficIn(&alr_detector, &timestamp_ms)
      .ForTimeMs(3000)
      .AtPercentOfEstimatedBitrate(100);
  EXPECT_FALSE(alr_detector.GetApplicationLimitedRegionStartTime());
}

TEST(AlrDetectorTest, BandwidthEstimateChanges) {
  FieldTrialBasedConfig field_trials;
  int64_t timestamp_ms = 1000;
  AlrDetector alr_detector(&field_trials);
  alr_detector.SetEstimatedBitrate(kEstimatedBitrateBps);

  // Start in non-ALR state.
  EXPECT_FALSE(alr_detector.GetApplicationLimitedRegionStartTime());

  // ALR starts when bitrate drops below 20%.
  SimulateOutgoingTrafficIn(&alr_detector, &timestamp_ms)
      .ForTimeMs(1000)
      .AtPercentOfEstimatedBitrate(20);
  EXPECT_TRUE(alr_detector.GetApplicationLimitedRegionStartTime());

  // When bandwidth estimate drops the detector should stay in ALR mode and quit
  // it shortly afterwards as the sender continues sending the same amount of
  // traffic. This is necessary to ensure that ProbeController can still react
  // to the BWE drop by initiating a new probe.
  alr_detector.SetEstimatedBitrate(kEstimatedBitrateBps / 5);
  EXPECT_TRUE(alr_detector.GetApplicationLimitedRegionStartTime());
  SimulateOutgoingTrafficIn(&alr_detector, &timestamp_ms)
      .ForTimeMs(1000)
      .AtPercentOfEstimatedBitrate(50);
  EXPECT_FALSE(alr_detector.GetApplicationLimitedRegionStartTime());
}

TEST(AlrDetectorTest, ParseControlFieldTrial) {
  webrtc::test::ScopedFieldTrials scoped_field_trial(
      "WebRTC-ProbingScreenshareBwe/Control/");
  std::optional<AlrExperimentSettings> parsed_params =
      AlrExperimentSettings::CreateFromFieldTrial(
          FieldTrialBasedConfig(), "WebRTC-ProbingScreenshareBwe");
  EXPECT_FALSE(static_cast<bool>(parsed_params));
}

TEST(AlrDetectorTest, ParseActiveFieldTrial) {
  webrtc::test::ScopedFieldTrials scoped_field_trial(
      "WebRTC-ProbingScreenshareBwe/1.1,2875,85,20,-20,1/");
  std::optional<AlrExperimentSettings> parsed_params =
      AlrExperimentSettings::CreateFromFieldTrial(
          FieldTrialBasedConfig(), "WebRTC-ProbingScreenshareBwe");
  ASSERT_TRUE(static_cast<bool>(parsed_params));
  EXPECT_EQ(1.1f, parsed_params->pacing_factor);
  EXPECT_EQ(2875, parsed_params->max_paced_queue_time);
  EXPECT_EQ(85, parsed_params->alr_bandwidth_usage_percent);
  EXPECT_EQ(20, parsed_params->alr_start_budget_level_percent);
  EXPECT_EQ(-20, parsed_params->alr_stop_budget_level_percent);
  EXPECT_EQ(1, parsed_params->group_id);
}

TEST(AlrDetectorTest, ParseAlrSpecificFieldTrial) {
  webrtc::test::ScopedFieldTrials scoped_field_trial(
      "WebRTC-AlrDetectorParameters/"
      "bw_usage:90%,start:0%,stop:-10%/");
  FieldTrialBasedConfig field_trials;
  AlrDetector alr_detector(&field_trials);
  int64_t timestamp_ms = 1000;
  alr_detector.SetEstimatedBitrate(kEstimatedBitrateBps);

  // Start in non-ALR state.
  EXPECT_FALSE(alr_detector.GetApplicationLimitedRegionStartTime());

  // ALR does not start at 100% utilization.
  SimulateOutgoingTrafficIn(&alr_detector, &timestamp_ms)
      .ForTimeMs(1000)
      .AtPercentOfEstimatedBitrate(100);
  EXPECT_FALSE(alr_detector.GetApplicationLimitedRegionStartTime());

  // ALR does start at 85% utilization.
  // Overused 10% above so it should take about 2s to reach a budget level of
  // 0%.
  SimulateOutgoingTrafficIn(&alr_detector, &timestamp_ms)
      .ForTimeMs(2100)
      .AtPercentOfEstimatedBitrate(85);
  EXPECT_TRUE(alr_detector.GetApplicationLimitedRegionStartTime());
}

}  // namespace webrtc
