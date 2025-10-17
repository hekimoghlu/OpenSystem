/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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
#include "modules/video_coding/codecs/test/videocodec_test_fixture_impl.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace test {

namespace {
// Codec settings.
const int kCifWidth = 352;
const int kCifHeight = 288;
const int kNumFrames = 100;

VideoCodecTestFixture::Config CreateConfig() {
  VideoCodecTestFixture::Config config;
  config.filename = "foreman_cif";
  config.filepath = ResourcePath(config.filename, "yuv");
  config.num_frames = kNumFrames;
  // Only allow encoder/decoder to use single core, for predictability.
  config.use_single_core = true;
  return config;
}
}  // namespace

TEST(VideoCodecTestOpenH264, ConstantHighBitrate) {
  auto frame_checker =
      std::make_unique<VideoCodecTestFixtureImpl::H264KeyframeChecker>();
  auto config = CreateConfig();
  config.SetCodecSettings(cricket::kH264CodecName, 1, 1, 1, false, true, false,
                          kCifWidth, kCifHeight);
  config.encoded_frame_checker = frame_checker.get();
  auto fixture = CreateVideoCodecTestFixture(config);

  std::vector<RateProfile> rate_profiles = {{500, 30, 0}};

  std::vector<RateControlThresholds> rc_thresholds = {
      {5, 1, 0, 0.1, 0.2, 0.1, 0, 1}};

  std::vector<QualityThresholds> quality_thresholds = {{37, 35, 0.93, 0.91}};

  fixture->RunTest(rate_profiles, &rc_thresholds, &quality_thresholds, nullptr);
}

// H264: Enable SingleNalUnit packetization mode. Encoder should split
// large frames into multiple slices and limit length of NAL units.
TEST(VideoCodecTestOpenH264, SingleNalUnit) {
  auto frame_checker =
      std::make_unique<VideoCodecTestFixtureImpl::H264KeyframeChecker>();
  auto config = CreateConfig();
  config.h264_codec_settings.packetization_mode =
      H264PacketizationMode::SingleNalUnit;
  config.max_payload_size_bytes = 500;
  config.SetCodecSettings(cricket::kH264CodecName, 1, 1, 1, false, true, false,
                          kCifWidth, kCifHeight);
  config.encoded_frame_checker = frame_checker.get();
  auto fixture = CreateVideoCodecTestFixture(config);

  std::vector<RateProfile> rate_profiles = {{500, 30, 0}};

  std::vector<RateControlThresholds> rc_thresholds = {
      {5, 1, 0, 0.1, 0.2, 0.1, 0, 1}};

  std::vector<QualityThresholds> quality_thresholds = {{37, 35, 0.93, 0.91}};

  BitstreamThresholds bs_thresholds = {config.max_payload_size_bytes};

  fixture->RunTest(rate_profiles, &rc_thresholds, &quality_thresholds,
                   &bs_thresholds);
}

}  // namespace test
}  // namespace webrtc
