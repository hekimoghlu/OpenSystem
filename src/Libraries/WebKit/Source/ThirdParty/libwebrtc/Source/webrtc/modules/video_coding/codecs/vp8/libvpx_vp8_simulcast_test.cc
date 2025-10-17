/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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
#include "modules/video_coding/codecs/vp8/include/vp8.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {

namespace {
std::unique_ptr<SimulcastTestFixture> CreateSpecificSimulcastTestFixture() {
  std::unique_ptr<VideoEncoderFactory> encoder_factory =
      std::make_unique<FunctionVideoEncoderFactory>(
          [](const Environment& env, const SdpVideoFormat& format) {
            return CreateVp8Encoder(env);
          });
  std::unique_ptr<VideoDecoderFactory> decoder_factory =
      std::make_unique<FunctionVideoDecoderFactory>(
          [](const Environment& env, const SdpVideoFormat& format) {
            return CreateVp8Decoder(env);
          });
  return CreateSimulcastTestFixture(std::move(encoder_factory),
                                    std::move(decoder_factory),
                                    SdpVideoFormat::VP8());
}
}  // namespace

TEST(LibvpxVp8SimulcastTest, TestKeyFrameRequestsOnAllStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestKeyFrameRequestsOnAllStreams();
}

TEST(LibvpxVp8SimulcastTest, TestKeyFrameRequestsOnSpecificStreams) {
  GTEST_SKIP() << "Not applicable to VP8.";
}

TEST(LibvpxVp8SimulcastTest, TestPaddingAllStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingAllStreams();
}

TEST(LibvpxVp8SimulcastTest, TestPaddingTwoStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingTwoStreams();
}

TEST(LibvpxVp8SimulcastTest, TestPaddingTwoStreamsOneMaxedOut) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingTwoStreamsOneMaxedOut();
}

TEST(LibvpxVp8SimulcastTest, TestPaddingOneStream) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingOneStream();
}

TEST(LibvpxVp8SimulcastTest, TestPaddingOneStreamTwoMaxedOut) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestPaddingOneStreamTwoMaxedOut();
}

TEST(LibvpxVp8SimulcastTest, TestSendAllStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSendAllStreams();
}

TEST(LibvpxVp8SimulcastTest, TestDisablingStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestDisablingStreams();
}

TEST(LibvpxVp8SimulcastTest, TestActiveStreams) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestActiveStreams();
}

TEST(LibvpxVp8SimulcastTest, TestSwitchingToOneStream) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSwitchingToOneStream();
}

TEST(LibvpxVp8SimulcastTest, TestSwitchingToOneOddStream) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSwitchingToOneOddStream();
}

TEST(LibvpxVp8SimulcastTest, TestSwitchingToOneSmallStream) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSwitchingToOneSmallStream();
}

TEST(LibvpxVp8SimulcastTest, TestSpatioTemporalLayers333PatternEncoder) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestSpatioTemporalLayers333PatternEncoder();
}

TEST(LibvpxVp8SimulcastTest, TestStrideEncodeDecode) {
  auto fixture = CreateSpecificSimulcastTestFixture();
  fixture->TestStrideEncodeDecode();
}

}  // namespace test
}  // namespace webrtc
