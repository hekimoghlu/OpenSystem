/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
#include "media/base/media_constants.h"
#include "modules/video_coding/codecs/test/objc_codec_factory_helper.h"
#include "modules/video_coding/codecs/test/videocodec_test_fixture_impl.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace test {

namespace {
const int kForemanNumFrames = 300;

VideoCodecTestFixture::Config CreateConfig() {
  VideoCodecTestFixture::Config config;
  config.filename = "foreman_cif";
  config.filepath = ResourcePath(config.filename, "yuv");
  config.num_frames = kForemanNumFrames;
  return config;
}

std::unique_ptr<VideoCodecTestFixture> CreateTestFixtureWithConfig(
    VideoCodecTestFixture::Config config) {
  auto decoder_factory = CreateObjCDecoderFactory();
  auto encoder_factory = CreateObjCEncoderFactory();
  return CreateVideoCodecTestFixture(config, std::move(decoder_factory),
                                     std::move(encoder_factory));
}
}  // namespace

// TODO(webrtc:9099): Disabled until the issue is fixed.
// HW codecs don't work on simulators. Only run these tests on device.
// #if TARGET_OS_IPHONE && !TARGET_IPHONE_SIMULATOR
// #define MAYBE_TEST TEST
// #else
#define MAYBE_TEST(s, name) TEST(s, DISABLED_##name)
// #endif

// TODO(kthelgason): Use RC Thresholds when the internal bitrateAdjuster is no
// longer in use.
MAYBE_TEST(VideoCodecTestVideoToolbox, ForemanCif500kbpsH264CBP) {
  const auto frame_checker =
      std::make_unique<VideoCodecTestFixtureImpl::H264KeyframeChecker>();
  auto config = CreateConfig();
  config.SetCodecSettings(cricket::kH264CodecName, 1, 1, 1, false, false, false,
                          352, 288);
  config.encoded_frame_checker = frame_checker.get();
  auto fixture = CreateTestFixtureWithConfig(config);

  std::vector<RateProfile> rate_profiles = {{500, 30, 0}};

  std::vector<QualityThresholds> quality_thresholds = {{33, 29, 0.9, 0.82}};

  fixture->RunTest(rate_profiles, nullptr, &quality_thresholds, nullptr);
}

MAYBE_TEST(VideoCodecTestVideoToolbox, ForemanCif500kbpsH264CHP) {
  const auto frame_checker =
      std::make_unique<VideoCodecTestFixtureImpl::H264KeyframeChecker>();
  auto config = CreateConfig();
  config.h264_codec_settings.profile = H264Profile::kProfileConstrainedHigh;
  config.SetCodecSettings(cricket::kH264CodecName, 1, 1, 1, false, false, false,
                          352, 288);
  config.encoded_frame_checker = frame_checker.get();
  auto fixture = CreateTestFixtureWithConfig(config);

  std::vector<RateProfile> rate_profiles = {{500, 30, 0}};

  std::vector<QualityThresholds> quality_thresholds = {{33, 30, 0.91, 0.83}};

  fixture->RunTest(rate_profiles, nullptr, &quality_thresholds, nullptr);
}

}  // namespace test
}  // namespace webrtc
