/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#include "api/test/create_videocodec_test_fixture.h"

#include <memory>
#include <utility>

#include "api/test/videocodec_test_fixture.h"
#include "api/video_codecs/video_decoder_factory.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "modules/video_coding/codecs/test/videocodec_test_fixture_impl.h"

namespace webrtc {
namespace test {

using Config = VideoCodecTestFixture::Config;

std::unique_ptr<VideoCodecTestFixture> CreateVideoCodecTestFixture(
    const Config& config) {
  return std::make_unique<VideoCodecTestFixtureImpl>(config);
}

std::unique_ptr<VideoCodecTestFixture> CreateVideoCodecTestFixture(
    const Config& config,
    std::unique_ptr<VideoDecoderFactory> decoder_factory,
    std::unique_ptr<VideoEncoderFactory> encoder_factory) {
  return std::make_unique<VideoCodecTestFixtureImpl>(
      config, std::move(decoder_factory), std::move(encoder_factory));
}

}  // namespace test
}  // namespace webrtc
