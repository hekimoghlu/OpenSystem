/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 8, 2024.
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
#ifndef API_TEST_MOCK_ENCODER_SELECTOR_H_
#define API_TEST_MOCK_ENCODER_SELECTOR_H_

#include <optional>

#include "api/units/data_rate.h"
#include "api/video/render_resolution.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "test/gmock.h"

namespace webrtc {

class MockEncoderSelector
    : public VideoEncoderFactory::EncoderSelectorInterface {
 public:
  MOCK_METHOD(void,
              OnCurrentEncoder,
              (const SdpVideoFormat& format),
              (override));

  MOCK_METHOD(std::optional<SdpVideoFormat>,
              OnAvailableBitrate,
              (const DataRate& rate),
              (override));

  MOCK_METHOD(std::optional<SdpVideoFormat>,
              OnResolutionChange,
              (const RenderResolution& resolution),
              (override));

  MOCK_METHOD(std::optional<SdpVideoFormat>, OnEncoderBroken, (), (override));
};

}  // namespace webrtc

#endif  // API_TEST_MOCK_ENCODER_SELECTOR_H_
