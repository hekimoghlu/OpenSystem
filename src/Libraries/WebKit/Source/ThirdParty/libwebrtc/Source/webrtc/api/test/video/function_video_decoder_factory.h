/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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
#ifndef API_TEST_VIDEO_FUNCTION_VIDEO_DECODER_FACTORY_H_
#define API_TEST_VIDEO_FUNCTION_VIDEO_DECODER_FACTORY_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "api/environment/environment.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_decoder.h"
#include "api/video_codecs/video_decoder_factory.h"

namespace webrtc {
namespace test {

// A decoder factory producing decoders by calling a supplied create function.
class FunctionVideoDecoderFactory final : public VideoDecoderFactory {
 public:
  explicit FunctionVideoDecoderFactory(
      std::function<std::unique_ptr<VideoDecoder>()> create)
      : create_([create = std::move(create)](const Environment&,
                                             const SdpVideoFormat&) {
          return create();
        }) {}
  explicit FunctionVideoDecoderFactory(
      std::function<std::unique_ptr<VideoDecoder>(const Environment&,
                                                  const SdpVideoFormat&)>
          create)
      : create_(std::move(create)) {}
  FunctionVideoDecoderFactory(
      std::function<std::unique_ptr<VideoDecoder>()> create,
      std::vector<SdpVideoFormat> sdp_video_formats)
      : create_([create = std::move(create)](const Environment&,
                                             const SdpVideoFormat&) {
          return create();
        }),
        sdp_video_formats_(std::move(sdp_video_formats)) {}

  std::vector<SdpVideoFormat> GetSupportedFormats() const override {
    return sdp_video_formats_;
  }

  std::unique_ptr<VideoDecoder> Create(const Environment& env,
                                       const SdpVideoFormat& format) override {
    return create_(env, format);
  }

 private:
  const std::function<std::unique_ptr<VideoDecoder>(const Environment& env,
                                                    const SdpVideoFormat&)>
      create_;
  const std::vector<SdpVideoFormat> sdp_video_formats_;
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_VIDEO_FUNCTION_VIDEO_DECODER_FACTORY_H_
