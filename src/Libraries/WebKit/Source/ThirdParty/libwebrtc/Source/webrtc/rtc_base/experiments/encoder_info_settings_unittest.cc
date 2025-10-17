/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
#include "rtc_base/experiments/encoder_info_settings.h"

#include "rtc_base/gunit.h"
#include "test/explicit_key_value_config.h"
#include "test/gmock.h"

namespace webrtc {

using test::ExplicitKeyValueConfig;

TEST(SimulcastEncoderAdapterSettingsTest, NoValuesWithoutFieldTrial) {
  ExplicitKeyValueConfig field_trials("");

  SimulcastEncoderAdapterEncoderInfoSettings settings(field_trials);
  EXPECT_EQ(std::nullopt, settings.requested_resolution_alignment());
  EXPECT_FALSE(settings.apply_alignment_to_all_simulcast_layers());
  EXPECT_TRUE(settings.resolution_bitrate_limits().empty());
}

TEST(SimulcastEncoderAdapterSettingsTest, NoValueForInvalidAlignment) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-SimulcastEncoderAdapter-GetEncoderInfoOverride/"
      "requested_resolution_alignment:0/");

  SimulcastEncoderAdapterEncoderInfoSettings settings(field_trials);
  EXPECT_EQ(std::nullopt, settings.requested_resolution_alignment());
}

TEST(SimulcastEncoderAdapterSettingsTest, GetResolutionAlignment) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-SimulcastEncoderAdapter-GetEncoderInfoOverride/"
      "requested_resolution_alignment:2/");

  SimulcastEncoderAdapterEncoderInfoSettings settings(field_trials);
  EXPECT_EQ(2u, settings.requested_resolution_alignment());
  EXPECT_FALSE(settings.apply_alignment_to_all_simulcast_layers());
  EXPECT_TRUE(settings.resolution_bitrate_limits().empty());
}

TEST(SimulcastEncoderAdapterSettingsTest, GetApplyAlignment) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-SimulcastEncoderAdapter-GetEncoderInfoOverride/"
      "requested_resolution_alignment:3,"
      "apply_alignment_to_all_simulcast_layers/");

  SimulcastEncoderAdapterEncoderInfoSettings settings(field_trials);
  EXPECT_EQ(3u, settings.requested_resolution_alignment());
  EXPECT_TRUE(settings.apply_alignment_to_all_simulcast_layers());
  EXPECT_TRUE(settings.resolution_bitrate_limits().empty());
}

TEST(SimulcastEncoderAdapterSettingsTest, GetResolutionBitrateLimits) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-SimulcastEncoderAdapter-GetEncoderInfoOverride/"
      "frame_size_pixels:123,"
      "min_start_bitrate_bps:11000,"
      "min_bitrate_bps:44000,"
      "max_bitrate_bps:77000/");

  SimulcastEncoderAdapterEncoderInfoSettings settings(field_trials);
  EXPECT_EQ(std::nullopt, settings.requested_resolution_alignment());
  EXPECT_FALSE(settings.apply_alignment_to_all_simulcast_layers());
  EXPECT_THAT(settings.resolution_bitrate_limits(),
              ::testing::ElementsAre(VideoEncoder::ResolutionBitrateLimits{
                  123, 11000, 44000, 77000}));
}

TEST(SimulcastEncoderAdapterSettingsTest, GetResolutionBitrateLimitsWithList) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-SimulcastEncoderAdapter-GetEncoderInfoOverride/"
      "frame_size_pixels:123|456|789,"
      "min_start_bitrate_bps:11000|22000|33000,"
      "min_bitrate_bps:44000|55000|66000,"
      "max_bitrate_bps:77000|88000|99000/");

  SimulcastEncoderAdapterEncoderInfoSettings settings(field_trials);
  EXPECT_THAT(
      settings.resolution_bitrate_limits(),
      ::testing::ElementsAre(
          VideoEncoder::ResolutionBitrateLimits{123, 11000, 44000, 77000},
          VideoEncoder::ResolutionBitrateLimits{456, 22000, 55000, 88000},
          VideoEncoder::ResolutionBitrateLimits{789, 33000, 66000, 99000}));
}

TEST(EncoderSettingsTest, CommonSettingsUsedIfEncoderNameUnspecified) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-VP8-GetEncoderInfoOverride/requested_resolution_alignment:2/"
      "WebRTC-GetEncoderInfoOverride/requested_resolution_alignment:3/");

  LibvpxVp8EncoderInfoSettings vp8_settings(field_trials);
  EXPECT_EQ(2u, vp8_settings.requested_resolution_alignment());
  LibvpxVp9EncoderInfoSettings vp9_settings(field_trials);
  EXPECT_EQ(3u, vp9_settings.requested_resolution_alignment());
}

}  // namespace webrtc
