/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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
#include "api/audio_codecs/builtin_audio_decoder_factory.h"

#include <memory>
#include <optional>
#include <vector>

#include "api/audio_codecs/L16/audio_decoder_L16.h"
#include "api/audio_codecs/audio_codec_pair_id.h"
#include "api/audio_codecs/audio_decoder.h"
#include "api/audio_codecs/audio_decoder_factory.h"
#include "api/audio_codecs/audio_decoder_factory_template.h"
#include "api/audio_codecs/audio_format.h"
#include "api/audio_codecs/g711/audio_decoder_g711.h"
#include "api/audio_codecs/g722/audio_decoder_g722.h"
#include "api/scoped_refptr.h"
#if WEBRTC_USE_BUILTIN_OPUS
#include "api/audio_codecs/opus/audio_decoder_multi_channel_opus.h"
#include "api/audio_codecs/opus/audio_decoder_opus.h"  // nogncheck
#endif

namespace webrtc {

namespace {

// Modify an audio decoder to not advertise support for anything.
template <typename T>
struct NotAdvertised {
  using Config = typename T::Config;
  static std::optional<Config> SdpToConfig(const SdpAudioFormat& audio_format) {
    return T::SdpToConfig(audio_format);
  }
  static void AppendSupportedDecoders(
      std::vector<AudioCodecSpec>* /* specs */) {
    // Don't advertise support for anything.
  }
  static std::unique_ptr<AudioDecoder> MakeAudioDecoder(
      const Config& config,
      std::optional<AudioCodecPairId> codec_pair_id = std::nullopt) {
    return T::MakeAudioDecoder(config, codec_pair_id);
  }
};

}  // namespace

rtc::scoped_refptr<AudioDecoderFactory> CreateBuiltinAudioDecoderFactory() {
  return CreateAudioDecoderFactory<

#if WEBRTC_USE_BUILTIN_OPUS
      AudioDecoderOpus, NotAdvertised<AudioDecoderMultiChannelOpus>,
#endif

      AudioDecoderG722,
      AudioDecoderG711, NotAdvertised<AudioDecoderL16>>();
}

}  // namespace webrtc
