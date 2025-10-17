/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#include <memory>
#include <vector>

#include "api/test/create_videocodec_test_fixture.h"
#include "api/test/video/function_video_encoder_factory.h"
#include "api/video_codecs/sdp_video_format.h"
#include "media/base/media_constants.h"
#include "media/engine/internal_decoder_factory.h"
#include "media/engine/internal_encoder_factory.h"
#include "media/engine/simulcast_encoder_adapter.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace test {
namespace {
// Test clips settings.
constexpr int kCifWidth = 352;
constexpr int kCifHeight = 288;
constexpr int kNumFramesLong = 300;

VideoCodecTestFixture::Config CreateConfig(std::string filename) {
  VideoCodecTestFixture::Config config;
  config.filename = filename;
  config.filepath = ResourcePath(config.filename, "yuv");
  config.num_frames = kNumFramesLong;
  config.use_single_core = true;
  return config;
}

TEST(VideoCodecTestAv1, HighBitrate) {
  auto config = CreateConfig("foreman_cif");
  config.SetCodecSettings(cricket::kAv1CodecName, 1, 1, 1, false, true, true,
                          kCifWidth, kCifHeight);
  config.codec_settings.SetScalabilityMode(ScalabilityMode::kL1T1);
  config.num_frames = kNumFramesLong;
  auto fixture = CreateVideoCodecTestFixture(config);

  std::vector<RateProfile> rate_profiles = {{500, 30, 0}};

  std::vector<RateControlThresholds> rc_thresholds = {
      {12, 1, 0, 1, 0.3, 0.1, 0, 1}};

  std::vector<QualityThresholds> quality_thresholds = {{37, 34, 0.94, 0.91}};

  fixture->RunTest(rate_profiles, &rc_thresholds, &quality_thresholds, nullptr);
}

TEST(VideoCodecTestAv1, VeryLowBitrate) {
  auto config = CreateConfig("foreman_cif");
  config.SetCodecSettings(cricket::kAv1CodecName, 1, 1, 1, false, true, true,
                          kCifWidth, kCifHeight);
  config.codec_settings.SetScalabilityMode(ScalabilityMode::kL1T1);
  auto fixture = CreateVideoCodecTestFixture(config);

  std::vector<RateProfile> rate_profiles = {{50, 30, 0}};

  std::vector<RateControlThresholds> rc_thresholds = {
      {15, 8, 75, 2, 2, 2, 2, 1}};

  std::vector<QualityThresholds> quality_thresholds = {{28, 24.8, 0.70, 0.55}};

  fixture->RunTest(rate_profiles, &rc_thresholds, &quality_thresholds, nullptr);
}

#if !defined(WEBRTC_ANDROID)
constexpr int kHdWidth = 1280;
constexpr int kHdHeight = 720;
TEST(VideoCodecTestAv1, Hd) {
  auto config = CreateConfig("ConferenceMotion_1280_720_50");
  config.SetCodecSettings(cricket::kAv1CodecName, 1, 1, 1, false, true, true,
                          kHdWidth, kHdHeight);
  config.codec_settings.SetScalabilityMode(ScalabilityMode::kL1T1);
  config.num_frames = kNumFramesLong;
  auto fixture = CreateVideoCodecTestFixture(config);

  std::vector<RateProfile> rate_profiles = {{1000, 50, 0}};

  std::vector<RateControlThresholds> rc_thresholds = {
      {13, 3, 0, 1, 0.3, 0.1, 0, 1}};

  std::vector<QualityThresholds> quality_thresholds = {
      {35.9, 31.5, 0.925, 0.865}};

  fixture->RunTest(rate_profiles, &rc_thresholds, &quality_thresholds, nullptr);
}
#endif

}  // namespace
}  // namespace test
}  // namespace webrtc
