/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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
#ifndef TEST_VIDEO_DECODER_PROXY_FACTORY_H_
#define TEST_VIDEO_DECODER_PROXY_FACTORY_H_

#include <memory>
#include <vector>

#include "api/video_codecs/video_decoder.h"
#include "api/video_codecs/video_decoder_factory.h"

namespace webrtc {
namespace test {

// A decoder factory with a single underlying VideoDecoder object, intended for
// test purposes. Each call to CreateVideoDecoder returns a proxy for the same
// decoder, typically an instance of FakeDecoder or MockEncoder.
class VideoDecoderProxyFactory final : public VideoDecoderFactory {
 public:
  explicit VideoDecoderProxyFactory(VideoDecoder* decoder)
      : decoder_(decoder) {}

  // Unused by tests.
  std::vector<SdpVideoFormat> GetSupportedFormats() const override {
    RTC_DCHECK_NOTREACHED();
    return {};
  }

  std::unique_ptr<VideoDecoder> Create(const Environment& env,
                                       const SdpVideoFormat& format) override {
    return std::make_unique<DecoderProxy>(decoder_);
  }

 private:
  // Wrapper class, since CreateVideoDecoder needs to surrender
  // ownership to the object it returns.
  class DecoderProxy final : public VideoDecoder {
   public:
    explicit DecoderProxy(VideoDecoder* decoder) : decoder_(decoder) {}

   private:
    int32_t Decode(const EncodedImage& input_image,
                   int64_t render_time_ms) override {
      return decoder_->Decode(input_image, render_time_ms);
    }
    bool Configure(const Settings& settings) override {
      return decoder_->Configure(settings);
    }
    int32_t RegisterDecodeCompleteCallback(
        DecodedImageCallback* callback) override {
      return decoder_->RegisterDecodeCompleteCallback(callback);
    }
    int32_t Release() override { return decoder_->Release(); }
    DecoderInfo GetDecoderInfo() const override {
      return decoder_->GetDecoderInfo();
    }
    const char* ImplementationName() const override {
      return decoder_->ImplementationName();
    }

    VideoDecoder* const decoder_;
  };

  VideoDecoder* const decoder_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_VIDEO_DECODER_PROXY_FACTORY_H_
