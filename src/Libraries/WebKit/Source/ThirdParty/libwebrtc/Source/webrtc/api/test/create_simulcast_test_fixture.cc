/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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
#include "api/test/create_simulcast_test_fixture.h"

#include <memory>
#include <utility>

#include "api/test/simulcast_test_fixture.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_decoder_factory.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "modules/video_coding/utility/simulcast_test_fixture_impl.h"

namespace webrtc {
namespace test {

std::unique_ptr<SimulcastTestFixture> CreateSimulcastTestFixture(
    std::unique_ptr<VideoEncoderFactory> encoder_factory,
    std::unique_ptr<VideoDecoderFactory> decoder_factory,
    SdpVideoFormat video_format) {
  return std::make_unique<SimulcastTestFixtureImpl>(
      std::move(encoder_factory), std::move(decoder_factory), video_format);
}

}  // namespace test
}  // namespace webrtc
