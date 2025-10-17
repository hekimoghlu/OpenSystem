/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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
#ifndef API_TEST_MOCK_VIDEO_DECODER_FACTORY_H_
#define API_TEST_MOCK_VIDEO_DECODER_FACTORY_H_

#include <memory>
#include <vector>

#include "api/environment/environment.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_decoder.h"
#include "api/video_codecs/video_decoder_factory.h"
#include "test/gmock.h"

namespace webrtc {

class MockVideoDecoderFactory : public VideoDecoderFactory {
 public:
  ~MockVideoDecoderFactory() override { Die(); }

  MOCK_METHOD(std::vector<SdpVideoFormat>,
              GetSupportedFormats,
              (),
              (const, override));
  MOCK_METHOD(std::unique_ptr<VideoDecoder>,
              Create,
              (const Environment&, const SdpVideoFormat&),
              (override));
  MOCK_METHOD(void, Die, ());
};
}  // namespace webrtc

#endif  // API_TEST_MOCK_VIDEO_DECODER_FACTORY_H_
