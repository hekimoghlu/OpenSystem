/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
#include "video/quality_convergence_controller.h"

#include <optional>

#include "test/gtest.h"
#include "test/scoped_key_value_config.h"

namespace webrtc {
namespace {
constexpr int kVp8DefaultStaticQpThreshold = 15;

TEST(QualityConvergenceController, Singlecast) {
  test::ScopedKeyValueConfig field_trials;
  QualityConvergenceController controller;
  controller.Initialize(1, /*encoder_min_qp=*/std::nullopt, kVideoCodecVP8,
                        field_trials);

  EXPECT_FALSE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, kVp8DefaultStaticQpThreshold + 1,
      /*is_refresh_frame=*/false));
  EXPECT_TRUE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, kVp8DefaultStaticQpThreshold,
      /*is_refresh_frame=*/false));
}

TEST(QualityConvergenceController, Simulcast) {
  test::ScopedKeyValueConfig field_trials;
  QualityConvergenceController controller;
  controller.Initialize(2, /*encoder_min_qp=*/std::nullopt, kVideoCodecVP8,
                        field_trials);

  EXPECT_FALSE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, kVp8DefaultStaticQpThreshold + 1,
      /*is_refresh_frame=*/false));
  EXPECT_FALSE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/1, kVp8DefaultStaticQpThreshold + 1,
      /*is_refresh_frame=*/false));

  // Layer 0 reaches target quality.
  EXPECT_TRUE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, kVp8DefaultStaticQpThreshold,
      /*is_refresh_frame=*/false));
  EXPECT_FALSE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/1, kVp8DefaultStaticQpThreshold + 1,
      /*is_refresh_frame=*/false));

  // Frames are repeated for both layers. Layer 0 still at target quality.
  EXPECT_TRUE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, kVp8DefaultStaticQpThreshold,
      /*is_refresh_frame=*/true));
  EXPECT_FALSE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/1, kVp8DefaultStaticQpThreshold + 1,
      /*is_refresh_frame=*/true));
}

TEST(QualityConvergenceController, InvalidLayerIndex) {
  test::ScopedKeyValueConfig field_trials;
  QualityConvergenceController controller;
  controller.Initialize(2, /*encoder_min_qp=*/std::nullopt, kVideoCodecVP8,
                        field_trials);

  EXPECT_FALSE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/-1, kVp8DefaultStaticQpThreshold,
      /*is_refresh_frame=*/false));
  EXPECT_FALSE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/3, kVp8DefaultStaticQpThreshold,
      /*is_refresh_frame=*/false));
}

TEST(QualityConvergenceController, UseMaxOfEncoderMinAndDefaultQpThresholds) {
  test::ScopedKeyValueConfig field_trials;
  QualityConvergenceController controller;
  controller.Initialize(1, kVp8DefaultStaticQpThreshold + 1, kVideoCodecVP8,
                        field_trials);

  EXPECT_FALSE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, kVp8DefaultStaticQpThreshold + 2,
      /*is_refresh_frame=*/false));
  EXPECT_TRUE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, kVp8DefaultStaticQpThreshold + 1,
      /*is_refresh_frame=*/false));
}

TEST(QualityConvergenceController, OverrideVp8StaticThreshold) {
  test::ScopedKeyValueConfig field_trials(
      "WebRTC-QCM-Static-VP8/static_qp_threshold:22/");
  QualityConvergenceController controller;
  controller.Initialize(1, /*encoder_min_qp=*/std::nullopt, kVideoCodecVP8,
                        field_trials);

  EXPECT_FALSE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, /*qp=*/23, /*is_refresh_frame=*/false));
  EXPECT_TRUE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, /*qp=*/22, /*is_refresh_frame=*/false));
}

TEST(QualityConvergenceMonitorSetup, OverrideVp9StaticThreshold) {
  test::ScopedKeyValueConfig field_trials(
      "WebRTC-QCM-Static-VP9/static_qp_threshold:44/");
  QualityConvergenceController controller;
  controller.Initialize(1, /*encoder_min_qp=*/std::nullopt, kVideoCodecVP9,
                        field_trials);

  EXPECT_FALSE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, /*qp=*/45, /*is_refresh_frame=*/false));
  EXPECT_TRUE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, /*qp=*/44, /*is_refresh_frame=*/false));
}

TEST(QualityConvergenceMonitorSetup, OverrideAv1StaticThreshold) {
  test::ScopedKeyValueConfig field_trials(
      "WebRTC-QCM-Static-AV1/static_qp_threshold:46/");
  QualityConvergenceController controller;
  controller.Initialize(1, /*encoder_min_qp=*/std::nullopt, kVideoCodecAV1,
                        field_trials);

  EXPECT_FALSE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, /*qp=*/47, /*is_refresh_frame=*/false));
  EXPECT_TRUE(controller.AddSampleAndCheckTargetQuality(
      /*layer_index=*/0, /*qp=*/46, /*is_refresh_frame=*/false));
}

}  // namespace
}  // namespace webrtc
