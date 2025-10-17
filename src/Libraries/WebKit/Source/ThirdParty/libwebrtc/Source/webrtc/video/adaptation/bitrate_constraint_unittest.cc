/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#include "video/adaptation/bitrate_constraint.h"

#include <utility>
#include <vector>

#include "api/video_codecs/scalability_mode.h"
#include "api/video_codecs/video_encoder.h"
#include "call/adaptation/encoder_settings.h"
#include "call/adaptation/test/fake_frame_rate_provider.h"
#include "call/adaptation/video_source_restrictions.h"
#include "call/adaptation/video_stream_input_state_provider.h"
#include "test/gtest.h"

namespace webrtc {

namespace {
const VideoSourceRestrictions k180p{/*max_pixels_per_frame=*/320 * 180,
                                    /*target_pixels_per_frame=*/320 * 180,
                                    /*max_frame_rate=*/30};
const VideoSourceRestrictions k360p{/*max_pixels_per_frame=*/640 * 360,
                                    /*target_pixels_per_frame=*/640 * 360,
                                    /*max_frame_rate=*/30};
const VideoSourceRestrictions k720p{/*max_pixels_per_frame=*/1280 * 720,
                                    /*target_pixels_per_frame=*/1280 * 720,
                                    /*max_frame_rate=*/30};

struct TestParams {
  bool active;
  std::optional<ScalabilityMode> scalability_mode;
};

void FillCodecConfig(VideoCodec* video_codec,
                     VideoEncoderConfig* encoder_config,
                     int width_px,
                     int height_px,
                     const std::vector<TestParams>& params,
                     bool svc) {
  size_t num_layers = params.size();
  video_codec->codecType = kVideoCodecVP8;
  video_codec->numberOfSimulcastStreams = svc ? 1 : num_layers;

  encoder_config->number_of_streams = svc ? 1 : num_layers;
  encoder_config->simulcast_layers.resize(num_layers);

  for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    int layer_width_px = width_px >> (num_layers - 1 - layer_idx);
    int layer_height_px = height_px >> (num_layers - 1 - layer_idx);

    if (params[layer_idx].scalability_mode)
      video_codec->SetScalabilityMode(*params[layer_idx].scalability_mode);
    video_codec->simulcastStream[layer_idx].active = params[layer_idx].active;
    video_codec->simulcastStream[layer_idx].width = layer_width_px;
    video_codec->simulcastStream[layer_idx].height = layer_height_px;

    encoder_config->simulcast_layers[layer_idx].scalability_mode =
        params[layer_idx].scalability_mode;
    encoder_config->simulcast_layers[layer_idx].active =
        params[layer_idx].active;
    encoder_config->simulcast_layers[layer_idx].width = layer_width_px;
    encoder_config->simulcast_layers[layer_idx].height = layer_height_px;
  }
}

constexpr int kStartBitrateBps360p = 500000;
constexpr int kStartBitrateBps720p = 1000000;

VideoEncoder::EncoderInfo MakeEncoderInfo() {
  VideoEncoder::EncoderInfo encoder_info;
  encoder_info.resolution_bitrate_limits = {
      {640 * 360, kStartBitrateBps360p, 0, 5000000},
      {1280 * 720, kStartBitrateBps720p, 0, 5000000},
      {1920 * 1080, 2000000, 0, 5000000}};
  return encoder_info;
}

}  // namespace

class BitrateConstraintTest : public ::testing::Test {
 public:
  BitrateConstraintTest()
      : frame_rate_provider_(), input_state_provider_(&frame_rate_provider_) {}

 protected:
  void OnEncoderSettingsUpdated(int width_px,
                                int height_px,
                                const std::vector<TestParams>& params,
                                bool svc = false) {
    VideoCodec video_codec;
    VideoEncoderConfig encoder_config;
    FillCodecConfig(&video_codec, &encoder_config, width_px, height_px, params,
                    svc);

    EncoderSettings encoder_settings(MakeEncoderInfo(),
                                     std::move(encoder_config), video_codec);
    bitrate_constraint_.OnEncoderSettingsUpdated(encoder_settings);
    input_state_provider_.OnEncoderSettingsChanged(encoder_settings);
  }

  FakeFrameRateProvider frame_rate_provider_;
  VideoStreamInputStateProvider input_state_provider_;
  BitrateConstraint bitrate_constraint_;
};

TEST_F(BitrateConstraintTest, AdaptUpAllowedAtSinglecastIfBitrateIsEnough) {
  OnEncoderSettingsUpdated(/*width_px=*/640, /*height_px=*/360,
                           {{.active = true}});

  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps720p);

  EXPECT_TRUE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k360p,
      /*restrictions_after=*/k720p));
}

TEST_F(BitrateConstraintTest,
       AdaptUpDisallowedAtSinglecastIfBitrateIsNotEnough) {
  OnEncoderSettingsUpdated(/*width_px=*/640, /*height_px=*/360,
                           {{.active = true}});

  // 1 bps less than needed for 720p.
  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps720p - 1);

  EXPECT_FALSE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k360p,
      /*restrictions_after=*/k720p));
}

TEST_F(BitrateConstraintTest,
       AdaptUpAllowedAtSinglecastIfBitrateIsEnoughForOneSpatialLayer) {
  OnEncoderSettingsUpdated(
      /*width_px=*/640, /*height_px=*/360,
      {{.active = true, .scalability_mode = ScalabilityMode::kL1T1}});

  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps720p);

  EXPECT_TRUE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k360p,
      /*restrictions_after=*/k720p));
}

TEST_F(BitrateConstraintTest,
       AdaptUpDisallowedAtSinglecastIfBitrateIsNotEnoughForOneSpatialLayer) {
  OnEncoderSettingsUpdated(
      /*width_px=*/640, /*height_px=*/360,
      {{.active = true, .scalability_mode = ScalabilityMode::kL1T1}});

  // 1 bps less than needed for 720p.
  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps720p - 1);

  EXPECT_FALSE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k360p,
      /*restrictions_after=*/k720p));
}

TEST_F(BitrateConstraintTest,
       AdaptUpAllowedAtSinglecastIfBitrateIsNotEnoughForMultipleSpatialLayers) {
  OnEncoderSettingsUpdated(
      /*width_px=*/640, /*height_px=*/360,
      {{.active = true, .scalability_mode = ScalabilityMode::kL2T1}});

  // 1 bps less than needed for 720p.
  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps720p - 1);

  EXPECT_TRUE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k360p,
      /*restrictions_after=*/k720p));
}

TEST_F(BitrateConstraintTest,
       AdaptUpAllowedAtSinglecastUpperLayerActiveIfBitrateIsEnough) {
  OnEncoderSettingsUpdated(
      /*width_px=*/640, /*height_px=*/360,
      {{.active = false, .scalability_mode = ScalabilityMode::kL2T1},
       {.active = true}});

  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps720p);

  EXPECT_TRUE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k360p,
      /*restrictions_after=*/k720p));
}

TEST_F(BitrateConstraintTest,
       AdaptUpDisallowedAtSinglecastUpperLayerActiveIfBitrateIsNotEnough) {
  OnEncoderSettingsUpdated(
      /*width_px=*/640, /*height_px=*/360,
      {{.active = false, .scalability_mode = ScalabilityMode::kL2T1},
       {.active = true}});

  // 1 bps less than needed for 720p.
  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps720p - 1);

  EXPECT_FALSE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k360p,
      /*restrictions_after=*/k720p));
}

TEST_F(BitrateConstraintTest, AdaptUpAllowedLowestActiveIfBitrateIsNotEnough) {
  OnEncoderSettingsUpdated(/*width_px=*/640, /*height_px=*/360,
                           {{.active = true}, {.active = false}});

  // 1 bps less than needed for 360p.
  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps360p - 1);

  EXPECT_TRUE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k180p,
      /*restrictions_after=*/k360p));
}

TEST_F(BitrateConstraintTest,
       AdaptUpAllowedLowestActiveIfBitrateIsNotEnoughForOneSpatialLayer) {
  OnEncoderSettingsUpdated(
      /*width_px=*/640, /*height_px=*/360,
      {{.active = true, .scalability_mode = ScalabilityMode::kL1T2},
       {.active = false}});

  // 1 bps less than needed for 360p.
  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps360p - 1);

  EXPECT_TRUE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k180p,
      /*restrictions_after=*/k360p));
}

TEST_F(BitrateConstraintTest,
       AdaptUpAllowedLowestActiveIfBitrateIsEnoughForOneSpatialLayerSvc) {
  OnEncoderSettingsUpdated(
      /*width_px=*/640, /*height_px=*/360,
      {{.active = true, .scalability_mode = ScalabilityMode::kL1T1},
       {.active = false}},
      /*svc=*/true);

  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps360p);

  EXPECT_TRUE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k180p,
      /*restrictions_after=*/k360p));
}

TEST_F(BitrateConstraintTest,
       AdaptUpDisallowedLowestActiveIfBitrateIsNotEnoughForOneSpatialLayerSvc) {
  OnEncoderSettingsUpdated(
      /*width_px=*/640, /*height_px=*/360,
      {{.active = true, .scalability_mode = ScalabilityMode::kL1T1},
       {.active = false}},
      /*svc=*/true);

  // 1 bps less than needed for 360p.
  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps360p - 1);

  EXPECT_FALSE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k180p,
      /*restrictions_after=*/k360p));
}

TEST_F(BitrateConstraintTest,
       AdaptUpAllowedLowestActiveIfBitrateIsNotEnoughForTwoSpatialLayersSvc) {
  OnEncoderSettingsUpdated(
      /*width_px=*/640, /*height_px=*/360,
      {{.active = true, .scalability_mode = ScalabilityMode::kL2T1},
       {.active = false}},
      /*svc=*/true);

  // 1 bps less than needed for 360p.
  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps360p - 1);

  EXPECT_TRUE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k180p,
      /*restrictions_after=*/k360p));
}

TEST_F(BitrateConstraintTest, AdaptUpAllowedAtSimulcastIfBitrateIsNotEnough) {
  OnEncoderSettingsUpdated(/*width_px=*/640, /*height_px=*/360,
                           {{.active = true}, {.active = true}});

  // 1 bps less than needed for 720p.
  bitrate_constraint_.OnEncoderTargetBitrateUpdated(kStartBitrateBps720p - 1);

  EXPECT_TRUE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k360p,
      /*restrictions_after=*/k720p));
}

TEST_F(BitrateConstraintTest,
       AdaptUpInFpsAllowedAtNoResolutionIncreaseIfBitrateIsNotEnough) {
  OnEncoderSettingsUpdated(/*width_px=*/640, /*height_px=*/360,
                           {{.active = true}});

  bitrate_constraint_.OnEncoderTargetBitrateUpdated(1);

  EXPECT_TRUE(bitrate_constraint_.IsAdaptationUpAllowed(
      input_state_provider_.InputState(),
      /*restrictions_before=*/k360p,
      /*restrictions_after=*/k360p));
}

}  // namespace webrtc
