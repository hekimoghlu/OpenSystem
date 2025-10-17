/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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

#include "api/test/create_simulcast_test_fixture.h"
#include "api/test/simulcast_test_fixture.h"
#include "api/test/video/function_video_decoder_factory.h"
#include "api/test/video/function_video_encoder_factory.h"
#include "modules/video_coding/codecs/h264/include/h264.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {

namespace {
std::unique_ptr<SimulcastTestFixture> CreateSpecificSimulcastTestFixture() {
  std::unique_ptr<VideoEncoderFactory> encoder_factory =
      std::make_unique<FunctionVideoEncoderFactory>(
          [](const Environment& env, const SdpVideoFormat& format) {
            return CreateH264Encoder(env);
          });
  std::unique_ptr<VideoDecoderFactory> decoder_factory =
      std::make_unique<FunctionVideoDecoderFactory>(
          []() { return H264Decoder::Create(); });
  return CreateSimulcastTestFixture(std::move(encoder_factory),
                                    std::move(decoder_factory),
                                    SdpVideoFormat::H264());
}
}  // namespace

TEST(TestH264Simulcast, TestKeyFrameRequestsOnAllStreams) {
  GTEST_SKIP() << "Not applicable to H264.";
}

TEST(TestH264Simulcast, TestKeyFrameRequestsOnSpecificStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestKeyFrameRequestsOnSpecificStreams();
}

TEST(TestH264Simulcast, TestPaddingAllStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingAllStreams();
}

TEST(TestH264Simulcast, TestPaddingTwoStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingTwoStreams();
}

TEST(TestH264Simulcast, TestPaddingTwoStreamsOneMaxedOut) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingTwoStreamsOneMaxedOut();
}

TEST(TestH264Simulcast, TestPaddingOneStream) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingOneStream();
}

TEST(TestH264Simulcast, TestPaddingOneStreamTwoMaxedOut) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingOneStreamTwoMaxedOut();
}

TEST(TestH264Simulcast, TestSendAllStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSendAllStreams();
}

TEST(TestH264Simulcast, TestDisablingStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestDisablingStreams();
}

TEST(TestH264Simulcast, TestActiveStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestActiveStreams();
}

TEST(TestH264Simulcast, TestSwitchingToOneStream) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSwitchingToOneStream();
}

TEST(TestH264Simulcast, TestSwitchingToOneOddStream) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSwitchingToOneOddStream();
}

TEST(TestH264Simulcast, TestStrideEncodeDecode) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestStrideEncodeDecode();
}

TEST(TestH264Simulcast, TestSpatioTemporalLayers333PatternEncoder) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSpatioTemporalLayers333PatternEncoder();
}

}  // namespace test
}  // namespace webrtc
