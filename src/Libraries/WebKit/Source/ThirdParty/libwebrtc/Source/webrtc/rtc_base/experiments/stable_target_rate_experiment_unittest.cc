/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
#include "rtc_base/experiments/stable_target_rate_experiment.h"

#include "test/explicit_key_value_config.h"
#include "test/gtest.h"

namespace webrtc {

using test::ExplicitKeyValueConfig;

TEST(StableBweExperimentTest, Default) {
  ExplicitKeyValueConfig field_trials("");
  StableTargetRateExperiment config(field_trials);
  EXPECT_FALSE(config.IsEnabled());
  EXPECT_EQ(config.GetVideoHysteresisFactor(), 1.2);
  EXPECT_EQ(config.GetScreenshareHysteresisFactor(), 1.35);
}

TEST(StableBweExperimentTest, EnabledNoHysteresis) {
  ExplicitKeyValueConfig field_trials("WebRTC-StableTargetRate/enabled:true/");

  StableTargetRateExperiment config(field_trials);
  EXPECT_TRUE(config.IsEnabled());
  EXPECT_EQ(config.GetVideoHysteresisFactor(), 1.2);
  EXPECT_EQ(config.GetScreenshareHysteresisFactor(), 1.35);
}

TEST(StableBweExperimentTest, EnabledWithHysteresis) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-StableTargetRate/"
      "enabled:true,"
      "video_hysteresis_factor:1.1,"
      "screenshare_hysteresis_factor:1.2/");

  StableTargetRateExperiment config(field_trials);
  EXPECT_TRUE(config.IsEnabled());
  EXPECT_EQ(config.GetVideoHysteresisFactor(), 1.1);
  EXPECT_EQ(config.GetScreenshareHysteresisFactor(), 1.2);
}

TEST(StableBweExperimentTest, HysteresisOverrideVideoRateHystersis) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-StableTargetRate/"
      "enabled:true,"
      "video_hysteresis_factor:1.1,"
      "screenshare_hysteresis_factor:1.2/"
      "WebRTC-VideoRateControl/video_hysteresis:1.3,"
      "screenshare_hysteresis:1.4/");

  StableTargetRateExperiment config(field_trials);
  EXPECT_TRUE(config.IsEnabled());
  EXPECT_EQ(config.GetVideoHysteresisFactor(), 1.1);
  EXPECT_EQ(config.GetScreenshareHysteresisFactor(), 1.2);
}

}  // namespace webrtc
