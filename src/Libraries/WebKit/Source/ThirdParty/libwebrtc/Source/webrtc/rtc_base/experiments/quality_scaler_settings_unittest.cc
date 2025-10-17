/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#include "rtc_base/experiments/quality_scaler_settings.h"

#include "test/gtest.h"
#include "test/scoped_key_value_config.h"

namespace webrtc {
namespace {

TEST(QualityScalerSettingsTest, ValuesNotSetByDefault) {
  webrtc::test::ScopedKeyValueConfig field_trials("");
  const auto settings = QualityScalerSettings(field_trials);
  EXPECT_FALSE(settings.MinFrames());
  EXPECT_FALSE(settings.InitialScaleFactor());
  EXPECT_FALSE(settings.ScaleFactor());
  EXPECT_FALSE(settings.InitialBitrateIntervalMs());
  EXPECT_FALSE(settings.InitialBitrateFactor());
}

TEST(QualityScalerSettingsTest, ParseMinFrames) {
  test::ScopedKeyValueConfig field_trials(
      "WebRTC-Video-QualityScalerSettings/min_frames:100/");
  EXPECT_EQ(100, QualityScalerSettings(field_trials).MinFrames());
}

TEST(QualityScalerSettingsTest, ParseInitialScaleFactor) {
  test::ScopedKeyValueConfig field_trials(
      "WebRTC-Video-QualityScalerSettings/initial_scale_factor:1.5/");
  EXPECT_EQ(1.5, QualityScalerSettings(field_trials).InitialScaleFactor());
}

TEST(QualityScalerSettingsTest, ParseScaleFactor) {
  test::ScopedKeyValueConfig field_trials(
      "WebRTC-Video-QualityScalerSettings/scale_factor:1.1/");
  EXPECT_EQ(1.1, QualityScalerSettings(field_trials).ScaleFactor());
}

TEST(QualityScalerSettingsTest, ParseInitialBitrateInterval) {
  test::ScopedKeyValueConfig field_trials(
      "WebRTC-Video-QualityScalerSettings/initial_bitrate_interval_ms:1000/");
  EXPECT_EQ(1000,
            QualityScalerSettings(field_trials).InitialBitrateIntervalMs());
}

TEST(QualityScalerSettingsTest, ParseInitialBitrateFactor) {
  test::ScopedKeyValueConfig field_trials(
      "WebRTC-Video-QualityScalerSettings/initial_bitrate_factor:0.75/");
  EXPECT_EQ(0.75, QualityScalerSettings(field_trials).InitialBitrateFactor());
}

TEST(QualityScalerSettingsTest, ParseAll) {
  test::ScopedKeyValueConfig field_trials(
      "WebRTC-Video-QualityScalerSettings/"
      "min_frames:100,initial_scale_factor:1.5,scale_factor:0.9,"
      "initial_bitrate_interval_ms:5500,initial_bitrate_factor:0.7/");
  const auto settings = QualityScalerSettings(field_trials);
  EXPECT_EQ(100, settings.MinFrames());
  EXPECT_EQ(1.5, settings.InitialScaleFactor());
  EXPECT_EQ(0.9, settings.ScaleFactor());
  EXPECT_EQ(5500, settings.InitialBitrateIntervalMs());
  EXPECT_EQ(0.7, settings.InitialBitrateFactor());
}

TEST(QualityScalerSettingsTest, DoesNotParseIncorrectValue) {
  test::ScopedKeyValueConfig field_trials(
      "WebRTC-Video-QualityScalerSettings/"
      "min_frames:a,initial_scale_factor:b,scale_factor:c,"
      "initial_bitrate_interval_ms:d,initial_bitrate_factor:e/");
  const auto settings = QualityScalerSettings(field_trials);
  EXPECT_FALSE(settings.MinFrames());
  EXPECT_FALSE(settings.InitialScaleFactor());
  EXPECT_FALSE(settings.ScaleFactor());
  EXPECT_FALSE(settings.InitialBitrateIntervalMs());
  EXPECT_FALSE(settings.InitialBitrateFactor());
}

TEST(QualityScalerSettingsTest, DoesNotReturnTooSmallValue) {
  test::ScopedKeyValueConfig field_trials(
      "WebRTC-Video-QualityScalerSettings/"
      "min_frames:0,initial_scale_factor:0.0,scale_factor:0.0,"
      "initial_bitrate_interval_ms:-1,initial_bitrate_factor:0.0/");
  const auto settings = QualityScalerSettings(field_trials);
  EXPECT_FALSE(settings.MinFrames());
  EXPECT_FALSE(settings.InitialScaleFactor());
  EXPECT_FALSE(settings.ScaleFactor());
  EXPECT_FALSE(settings.InitialBitrateIntervalMs());
  EXPECT_FALSE(settings.InitialBitrateFactor());
}

}  // namespace
}  // namespace webrtc
