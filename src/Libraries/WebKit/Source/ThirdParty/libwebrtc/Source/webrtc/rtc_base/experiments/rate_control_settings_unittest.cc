/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 9, 2022.
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
#include "rtc_base/experiments/rate_control_settings.h"

#include "api/video_codecs/video_codec.h"
#include "test/explicit_key_value_config.h"
#include "test/gmock.h"
#include "test/gtest.h"
#include "video/config/video_encoder_config.h"

namespace webrtc {

namespace {

using test::ExplicitKeyValueConfig;
using ::testing::DoubleEq;
using ::testing::Optional;

RateControlSettings ParseFrom(absl::string_view field_trials) {
  return RateControlSettings(ExplicitKeyValueConfig(field_trials));
}

TEST(RateControlSettingsTest, CongestionWindow) {
  EXPECT_TRUE(ParseFrom("").UseCongestionWindow());

  const RateControlSettings settings =
      ParseFrom("WebRTC-CongestionWindow/QueueSize:100/");
  EXPECT_TRUE(settings.UseCongestionWindow());
  EXPECT_EQ(settings.GetCongestionWindowAdditionalTimeMs(), 100);
}

TEST(RateControlSettingsTest, CongestionWindowPushback) {
  EXPECT_TRUE(ParseFrom("").UseCongestionWindowPushback());

  const RateControlSettings settings =
      ParseFrom("WebRTC-CongestionWindow/QueueSize:100,MinBitrate:100000/");
  EXPECT_TRUE(settings.UseCongestionWindowPushback());
  EXPECT_EQ(settings.CongestionWindowMinPushbackTargetBitrateBps(), 100000u);
}

TEST(RateControlSettingsTest, CongestionWindowPushbackDropframe) {
  EXPECT_TRUE(ParseFrom("").UseCongestionWindowPushback());

  const RateControlSettings settings = ParseFrom(
      "WebRTC-CongestionWindow/"
      "QueueSize:100,MinBitrate:100000,DropFrame:true/");
  EXPECT_TRUE(settings.UseCongestionWindowPushback());
  EXPECT_EQ(settings.CongestionWindowMinPushbackTargetBitrateBps(), 100000u);
  EXPECT_TRUE(settings.UseCongestionWindowDropFrameOnly());
}

TEST(RateControlSettingsTest, CongestionWindowPushbackDefaultConfig) {
  const RateControlSettings settings = ParseFrom("");
  EXPECT_TRUE(settings.UseCongestionWindowPushback());
  EXPECT_EQ(settings.CongestionWindowMinPushbackTargetBitrateBps(), 30000u);
  EXPECT_TRUE(settings.UseCongestionWindowDropFrameOnly());
}

TEST(RateControlSettingsTest, PacingFactor) {
  EXPECT_FALSE(ParseFrom("").GetPacingFactor());

  EXPECT_THAT(
      ParseFrom("WebRTC-VideoRateControl/pacing_factor:1.2/").GetPacingFactor(),
      Optional(DoubleEq(1.2)));
}

TEST(RateControlSettingsTest, AlrProbing) {
  EXPECT_FALSE(ParseFrom("").UseAlrProbing());

  EXPECT_TRUE(
      ParseFrom("WebRTC-VideoRateControl/alr_probing:1/").UseAlrProbing());
}

TEST(RateControlSettingsTest, LibvpxVp8QpMax) {
  EXPECT_FALSE(ParseFrom("").LibvpxVp8QpMax());

  EXPECT_EQ(
      ParseFrom("WebRTC-VideoRateControl/vp8_qp_max:50/").LibvpxVp8QpMax(), 50);
}

TEST(RateControlSettingsTest, DoesNotGetTooLargeLibvpxVp8QpMaxValue) {
  EXPECT_FALSE(
      ParseFrom("WebRTC-VideoRateControl/vp8_qp_max:70/").LibvpxVp8QpMax());
}

TEST(RateControlSettingsTest, LibvpxVp8MinPixels) {
  EXPECT_FALSE(ParseFrom("").LibvpxVp8MinPixels());

  EXPECT_EQ(ParseFrom("WebRTC-VideoRateControl/vp8_min_pixels:50000/")
                .LibvpxVp8MinPixels(),
            50000);
}

TEST(RateControlSettingsTest, DoesNotGetTooSmallLibvpxVp8MinPixelValue) {
  EXPECT_FALSE(ParseFrom("WebRTC-VideoRateControl/vp8_min_pixels:0/")
                   .LibvpxVp8MinPixels());
}

TEST(RateControlSettingsTest, LibvpxTrustedRateController) {
  const RateControlSettings default_settings = ParseFrom("");
  EXPECT_TRUE(default_settings.LibvpxVp8TrustedRateController());
  EXPECT_TRUE(default_settings.LibvpxVp9TrustedRateController());

  const RateControlSettings settings =
      ParseFrom("WebRTC-VideoRateControl/trust_vp8:0,trust_vp9:0/");
  EXPECT_FALSE(settings.LibvpxVp8TrustedRateController());
  EXPECT_FALSE(settings.LibvpxVp9TrustedRateController());
}

TEST(RateControlSettingsTest, Vp8BaseHeavyTl3RateAllocationLegacyKey) {
  EXPECT_FALSE(ParseFrom("").Vp8BaseHeavyTl3RateAllocation());

  EXPECT_TRUE(ParseFrom("WebRTC-UseBaseHeavyVP8TL3RateAllocation/Enabled/")
                  .Vp8BaseHeavyTl3RateAllocation());
}

TEST(RateControlSettingsTest,
     Vp8BaseHeavyTl3RateAllocationVideoRateControlKey) {
  EXPECT_FALSE(ParseFrom("").Vp8BaseHeavyTl3RateAllocation());

  EXPECT_TRUE(ParseFrom("WebRTC-VideoRateControl/vp8_base_heavy_tl3_alloc:1/")
                  .Vp8BaseHeavyTl3RateAllocation());
}

TEST(RateControlSettingsTest,
     Vp8BaseHeavyTl3RateAllocationVideoRateControlKeyOverridesLegacyKey) {
  EXPECT_FALSE(ParseFrom("").Vp8BaseHeavyTl3RateAllocation());

  EXPECT_FALSE(ParseFrom("WebRTC-UseBaseHeavyVP8TL3RateAllocation/Enabled/"
                         "WebRTC-VideoRateControl/vp8_base_heavy_tl3_alloc:0/")
                   .Vp8BaseHeavyTl3RateAllocation());
}

TEST(RateControlSettingsTest, UseEncoderBitrateAdjuster) {
  EXPECT_TRUE(ParseFrom("").UseEncoderBitrateAdjuster());

  EXPECT_FALSE(ParseFrom("WebRTC-VideoRateControl/bitrate_adjuster:false/")
                   .UseEncoderBitrateAdjuster());
}

}  // namespace

}  // namespace webrtc
