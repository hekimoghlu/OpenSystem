/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
#ifndef TEST_VIDEO_ENCODER_NULLABLE_PROXY_FACTORY_H_
#define TEST_VIDEO_ENCODER_NULLABLE_PROXY_FACTORY_H_

#include <memory>
#include <vector>

#include "api/environment/environment.h"
#include "api/video_codecs/video_encoder.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "test/video_encoder_proxy_factory.h"

namespace webrtc {
namespace test {

class VideoEncoderNullableProxyFactory final : public VideoEncoderProxyFactory {
 public:
  explicit VideoEncoderNullableProxyFactory(
      VideoEncoder* encoder,
      EncoderSelectorInterface* encoder_selector)
      : VideoEncoderProxyFactory(encoder, encoder_selector) {}

  ~VideoEncoderNullableProxyFactory() override = default;

  std::unique_ptr<VideoEncoder> Create(const Environment& env,
                                       const SdpVideoFormat& format) override {
    if (!encoder_) {
      return nullptr;
    }
    return VideoEncoderProxyFactory::Create(env, format);
  }
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_VIDEO_ENCODER_NULLABLE_PROXY_FACTORY_H_
