/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
#include "test/fake_vp8_encoder.h"

#include <memory>
#include <utility>

#include "api/test/create_simulcast_test_fixture.h"
#include "api/test/simulcast_test_fixture.h"
#include "api/test/video/function_video_decoder_factory.h"
#include "api/test/video/function_video_encoder_factory.h"
#include "modules/video_coding/utility/simulcast_test_fixture_impl.h"
#include "test/fake_vp8_decoder.h"

namespace webrtc {
namespace test {

namespace {

std::unique_ptr<SimulcastTestFixture> CreateSpecificSimulcastTestFixture() {
  std::unique_ptr<VideoEncoderFactory> encoder_factory =
      std::make_unique<FunctionVideoEncoderFactory>(
          [](const Environment& env, const SdpVideoFormat& format) {
            return std::make_unique<FakeVp8Encoder>(env);
          });
  std::unique_ptr<VideoDecoderFactory> decoder_factory =
      std::make_unique<FunctionVideoDecoderFactory>(
          []() { return std::make_unique<FakeVp8Decoder>(); });
  return CreateSimulcastTestFixture(std::move(encoder_factory),
                                    std::move(decoder_factory),
                                    SdpVideoFormat::VP8());
}
}  // namespace

TEST(TestFakeVp8Codec, TestKeyFrameRequestsOnAllStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestKeyFrameRequestsOnAllStreams();
}

TEST(TestFakeVp8Codec, TestPaddingAllStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingAllStreams();
}

TEST(TestFakeVp8Codec, TestPaddingTwoStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingTwoStreams();
}

TEST(TestFakeVp8Codec, TestPaddingTwoStreamsOneMaxedOut) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingTwoStreamsOneMaxedOut();
}

TEST(TestFakeVp8Codec, TestPaddingOneStream) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingOneStream();
}

TEST(TestFakeVp8Codec, TestPaddingOneStreamTwoMaxedOut) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingOneStreamTwoMaxedOut();
}

TEST(TestFakeVp8Codec, TestSendAllStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSendAllStreams();
}

TEST(TestFakeVp8Codec, TestDisablingStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestDisablingStreams();
}

TEST(TestFakeVp8Codec, TestSwitchingToOneStream) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSwitchingToOneStream();
}

TEST(TestFakeVp8Codec, TestSwitchingToOneOddStream) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSwitchingToOneOddStream();
}

TEST(TestFakeVp8Codec, TestSwitchingToOneSmallStream) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSwitchingToOneSmallStream();
}

TEST(TestFakeVp8Codec, TestSpatioTemporalLayers333PatternEncoder) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSpatioTemporalLayers333PatternEncoder();
}

TEST(TestFakeVp8Codec, TestDecodeWidthHeightSet) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestDecodeWidthHeightSet();
}

TEST(TestFakeVp8Codec,
     TestEncoderInfoForDefaultTemporalLayerProfileHasFpsAllocation) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestEncoderInfoForDefaultTemporalLayerProfileHasFpsAllocation();
}

}  // namespace test
}  // namespace webrtc
