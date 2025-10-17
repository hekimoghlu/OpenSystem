/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
#ifndef API_TEST_VIDEO_FUNCTION_VIDEO_ENCODER_FACTORY_H_
#define API_TEST_VIDEO_FUNCTION_VIDEO_ENCODER_FACTORY_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "api/environment/environment.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_encoder.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

// An encoder factory producing encoders by calling a supplied create
// function.
class FunctionVideoEncoderFactory final : public VideoEncoderFactory {
 public:
  explicit FunctionVideoEncoderFactory(
      std::function<std::unique_ptr<VideoEncoder>()> create)
      : create_([create = std::move(create)](const Environment&,
                                             const SdpVideoFormat&) {
          return create();
        }) {}
  explicit FunctionVideoEncoderFactory(
      std::function<std::unique_ptr<VideoEncoder>(const Environment&,
                                                  const SdpVideoFormat&)>
          create)
      : create_(std::move(create)) {}

  // Unused by tests.
  std::vector<SdpVideoFormat> GetSupportedFormats() const override {
    RTC_DCHECK_NOTREACHED();
    return {};
  }

  std::unique_ptr<VideoEncoder> Create(const Environment& env,
                                       const SdpVideoFormat& format) override {
    return create_(env, format);
  }

 private:
  const std::function<std::unique_ptr<VideoEncoder>(const Environment&,
                                                    const SdpVideoFormat&)>
      create_;
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_VIDEO_FUNCTION_VIDEO_ENCODER_FACTORY_H_
