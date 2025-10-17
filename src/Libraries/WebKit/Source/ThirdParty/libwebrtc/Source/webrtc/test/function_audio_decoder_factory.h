/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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
#ifndef TEST_FUNCTION_AUDIO_DECODER_FACTORY_H_
#define TEST_FUNCTION_AUDIO_DECODER_FACTORY_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "api/audio_codecs/audio_decoder_factory.h"
#include "api/audio_codecs/audio_format.h"
#include "api/environment/environment.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

// A decoder factory producing decoders by calling a supplied create function.
class FunctionAudioDecoderFactory : public AudioDecoderFactory {
 public:
  explicit FunctionAudioDecoderFactory(
      std::function<std::unique_ptr<AudioDecoder>()> create)
      : create_([create](const Environment&,
                         const SdpAudioFormat&,
                         std::optional<AudioCodecPairId> codec_pair_id) {
          return create();
        }) {}
  explicit FunctionAudioDecoderFactory(
      std::function<std::unique_ptr<AudioDecoder>(
          const Environment&,
          const SdpAudioFormat&,
          std::optional<AudioCodecPairId> codec_pair_id)> create)
      : create_(std::move(create)) {}

  // Unused by tests.
  std::vector<AudioCodecSpec> GetSupportedDecoders() override {
    RTC_DCHECK_NOTREACHED();
    return {};
  }

  bool IsSupportedDecoder(const SdpAudioFormat& format) override {
    return true;
  }

  std::unique_ptr<AudioDecoder> Create(
      const Environment& env,
      const SdpAudioFormat& format,
      std::optional<AudioCodecPairId> codec_pair_id) override {
    return create_(env, format, codec_pair_id);
  }

 private:
  const std::function<std::unique_ptr<AudioDecoder>(
      const Environment&,
      const SdpAudioFormat&,
      std::optional<AudioCodecPairId> codec_pair_id)>
      create_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_FUNCTION_AUDIO_DECODER_FACTORY_H_
