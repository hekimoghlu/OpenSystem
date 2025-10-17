/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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
#include <stddef.h>

#include "api/test/videocodec_test_fixture.h"
#include "api/video_codecs/video_codec.h"
#include "test/gmock.h"
#include "test/gtest.h"
#include "test/video_codec_settings.h"

using ::testing::ElementsAre;

namespace webrtc {
namespace test {

using Config = VideoCodecTestFixture::Config;

namespace {
const size_t kNumTemporalLayers = 2;
}  // namespace

TEST(Config, NumberOfCoresWithUseSingleCore) {
  Config config;
  config.use_single_core = true;
  EXPECT_EQ(1u, config.NumberOfCores());
}

TEST(Config, NumberOfCoresWithoutUseSingleCore) {
  Config config;
  config.use_single_core = false;
  EXPECT_GE(config.NumberOfCores(), 1u);
}

TEST(Config, NumberOfTemporalLayersIsOne) {
  Config config;
  webrtc::test::CodecSettings(kVideoCodecH264, &config.codec_settings);
  EXPECT_EQ(1u, config.NumberOfTemporalLayers());
}

TEST(Config, NumberOfTemporalLayers_Vp8) {
  Config config;
  webrtc::test::CodecSettings(kVideoCodecVP8, &config.codec_settings);
  config.codec_settings.VP8()->numberOfTemporalLayers = kNumTemporalLayers;
  EXPECT_EQ(kNumTemporalLayers, config.NumberOfTemporalLayers());
}

TEST(Config, NumberOfTemporalLayers_Vp9) {
  Config config;
  webrtc::test::CodecSettings(kVideoCodecVP9, &config.codec_settings);
  config.codec_settings.VP9()->numberOfTemporalLayers = kNumTemporalLayers;
  EXPECT_EQ(kNumTemporalLayers, config.NumberOfTemporalLayers());
}

}  // namespace test
}  // namespace webrtc
