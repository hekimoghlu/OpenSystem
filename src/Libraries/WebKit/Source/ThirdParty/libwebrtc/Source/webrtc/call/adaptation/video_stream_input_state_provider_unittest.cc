/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#include "call/adaptation/video_stream_input_state_provider.h"

#include <utility>

#include "api/video_codecs/video_encoder.h"
#include "call/adaptation/encoder_settings.h"
#include "call/adaptation/test/fake_frame_rate_provider.h"
#include "test/gtest.h"

namespace webrtc {

TEST(VideoStreamInputStateProviderTest, DefaultValues) {
  FakeFrameRateProvider frame_rate_provider;
  VideoStreamInputStateProvider input_state_provider(&frame_rate_provider);
  VideoStreamInputState input_state = input_state_provider.InputState();
  EXPECT_EQ(false, input_state.has_input());
  EXPECT_EQ(std::nullopt, input_state.frame_size_pixels());
  EXPECT_EQ(0, input_state.frames_per_second());
  EXPECT_EQ(VideoCodecType::kVideoCodecGeneric, input_state.video_codec_type());
  EXPECT_EQ(kDefaultMinPixelsPerFrame, input_state.min_pixels_per_frame());
  EXPECT_EQ(std::nullopt, input_state.single_active_stream_pixels());
}

TEST(VideoStreamInputStateProviderTest, ValuesSet) {
  FakeFrameRateProvider frame_rate_provider;
  VideoStreamInputStateProvider input_state_provider(&frame_rate_provider);
  input_state_provider.OnHasInputChanged(true);
  input_state_provider.OnFrameSizeObserved(42);
  frame_rate_provider.set_fps(123);
  VideoEncoder::EncoderInfo encoder_info;
  encoder_info.scaling_settings.min_pixels_per_frame = 1337;
  VideoEncoderConfig encoder_config;
  encoder_config.codec_type = VideoCodecType::kVideoCodecVP9;
  VideoCodec video_codec;
  video_codec.codecType = VideoCodecType::kVideoCodecVP8;
  video_codec.numberOfSimulcastStreams = 2;
  video_codec.simulcastStream[0].active = false;
  video_codec.simulcastStream[1].active = true;
  video_codec.simulcastStream[1].width = 111;
  video_codec.simulcastStream[1].height = 222;
  input_state_provider.OnEncoderSettingsChanged(EncoderSettings(
      std::move(encoder_info), std::move(encoder_config), video_codec));
  VideoStreamInputState input_state = input_state_provider.InputState();
  EXPECT_EQ(true, input_state.has_input());
  EXPECT_EQ(42, input_state.frame_size_pixels());
  EXPECT_EQ(123, input_state.frames_per_second());
  EXPECT_EQ(VideoCodecType::kVideoCodecVP9, input_state.video_codec_type());
  EXPECT_EQ(1337, input_state.min_pixels_per_frame());
  EXPECT_EQ(111 * 222, input_state.single_active_stream_pixels());
}

}  // namespace webrtc
