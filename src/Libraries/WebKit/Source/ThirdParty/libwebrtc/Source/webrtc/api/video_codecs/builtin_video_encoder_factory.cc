/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
#include "api/video_codecs/builtin_video_encoder_factory.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "api/environment/environment.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_encoder.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "media/engine/internal_encoder_factory.h"
#include "media/engine/simulcast_encoder_adapter.h"

namespace webrtc {

namespace {

// This class wraps the internal factory and adds simulcast.
class BuiltinVideoEncoderFactory : public VideoEncoderFactory {
 public:
  BuiltinVideoEncoderFactory()
      : internal_encoder_factory_(new InternalEncoderFactory()) {}

  std::unique_ptr<VideoEncoder> Create(const Environment& env,
                                       const SdpVideoFormat& format) override {
    // Try creating an InternalEncoderFactory-backed SimulcastEncoderAdapter.
    // The adapter has a passthrough mode for the case that simulcast is not
    // used, so all responsibility can be delegated to it.
    if (format.IsCodecInList(
            internal_encoder_factory_->GetSupportedFormats())) {
      return std::make_unique<SimulcastEncoderAdapter>(
          env,
          /*primary_factory=*/internal_encoder_factory_.get(),
          /*fallback_factory=*/nullptr, format);
    }

    return nullptr;
  }

  std::vector<SdpVideoFormat> GetSupportedFormats() const override {
    return internal_encoder_factory_->GetSupportedFormats();
  }

  CodecSupport QueryCodecSupport(
      const SdpVideoFormat& format,
      std::optional<std::string> scalability_mode) const override {
    return internal_encoder_factory_->QueryCodecSupport(format,
                                                        scalability_mode);
  }

 private:
  const std::unique_ptr<VideoEncoderFactory> internal_encoder_factory_;
};

}  // namespace

std::unique_ptr<VideoEncoderFactory> CreateBuiltinVideoEncoderFactory() {
  return std::make_unique<BuiltinVideoEncoderFactory>();
}

}  // namespace webrtc
