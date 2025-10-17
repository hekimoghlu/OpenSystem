/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#include "rtc_base/experiments/min_video_bitrate_experiment.h"

#include <optional>

#include "api/units/data_rate.h"
#include "api/video/video_codec_type.h"
#include "test/explicit_key_value_config.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using test::ExplicitKeyValueConfig;

TEST(GetExperimentalMinVideoBitrateTest,
     NulloptForAllCodecsIfFieldTrialUndefined) {
  ExplicitKeyValueConfig field_trials("");

  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecGeneric),
            std::nullopt);
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP8),
            std::nullopt);
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP9),
            std::nullopt);
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecH264),
            std::nullopt);
}

TEST(GetExperimentalMinVideoBitrateTest,
     NulloptForAllCodecsIfFieldTrialDisabled) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-Video-MinVideoBitrate/Disabled,br:123kbps/");

  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecGeneric),
            std::nullopt);
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP8),
            std::nullopt);
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP9),
            std::nullopt);
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecH264),
            std::nullopt);
}

TEST(GetExperimentalMinVideoBitrateTest, BrForAllCodecsIfDefined) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-Video-MinVideoBitrate/Enabled,br:123kbps/");

  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecGeneric),
            DataRate::KilobitsPerSec(123));
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP8),
            DataRate::KilobitsPerSec(123));
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP9),
            DataRate::KilobitsPerSec(123));
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecH264),
            DataRate::KilobitsPerSec(123));
}

TEST(GetExperimentalMinVideoBitrateTest, BrTrumpsSpecificCodecConfigs) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-Video-MinVideoBitrate/"
      "Enabled,br:123kbps,vp8_br:100kbps,vp9_br:200kbps,h264_br:300kbps/");

  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecGeneric),
            DataRate::KilobitsPerSec(123));
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP8),
            DataRate::KilobitsPerSec(123));
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP9),
            DataRate::KilobitsPerSec(123));
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecH264),
            DataRate::KilobitsPerSec(123));
}

TEST(GetExperimentalMinVideoBitrateTest,
     SpecificCodecConfigsIgnoredIfExpDisabled) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-Video-MinVideoBitrate/"
      "Disabled,vp8_br:100kbps,vp9_br:200kbps,h264_br:300kbps/");

  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecGeneric),
            std::nullopt);
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP8),
            std::nullopt);
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP9),
            std::nullopt);
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecH264),
            std::nullopt);
}

TEST(GetExperimentalMinVideoBitrateTest, SpecificCodecConfigsUsedIfExpEnabled) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-Video-MinVideoBitrate/"
      "Enabled,vp8_br:100kbps,vp9_br:200kbps,h264_br:300kbps/");

  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecGeneric),
            std::nullopt);
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP8),
            DataRate::KilobitsPerSec(100));
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP9),
            DataRate::KilobitsPerSec(200));
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecH264),
            DataRate::KilobitsPerSec(300));
}

TEST(GetExperimentalMinVideoBitrateTest,
     Vp8BitrateValueTakenFromFallbackIfAvailable) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-Video-MinVideoBitrate/"
      "Enabled,vp8_br:100kbps,vp9_br:200kbps,h264_br:300kbps/"
      "WebRTC-VP8-Forced-Fallback-Encoder-v2/"
      "Enabled-444444,555555,666666/");

  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP8),
            DataRate::BitsPerSec(666666));
}

TEST(GetExperimentalMinVideoBitrateTest,
     NonVp8BitrateValuesTakenFromMinVideoBitrate) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-Video-MinVideoBitrate/"
      "Enabled,vp8_br:100kbps,vp9_br:200kbps,h264_br:300kbps/"
      "WebRTC-VP8-Forced-Fallback-Encoder-v2/"
      "Enabled-444444,555555,666666/");

  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecGeneric),
            std::nullopt);
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecVP9),
            DataRate::KilobitsPerSec(200));
  EXPECT_EQ(GetExperimentalMinVideoBitrate(field_trials, kVideoCodecH264),
            DataRate::KilobitsPerSec(300));
}

}  // namespace
}  // namespace webrtc
