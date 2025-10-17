/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 1, 2022.
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
#ifndef API_AUDIO_CODECS_OPUS_AUDIO_DECODER_OPUS_H_
#define API_AUDIO_CODECS_OPUS_AUDIO_DECODER_OPUS_H_

#include <memory>
#include <optional>
#include <vector>

#include "api/audio_codecs/audio_codec_pair_id.h"
#include "api/audio_codecs/audio_decoder.h"
#include "api/audio_codecs/audio_format.h"
#include "api/environment/environment.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Opus decoder API for use as a template parameter to
// CreateAudioDecoderFactory<...>().
struct RTC_EXPORT AudioDecoderOpus {
  struct Config {
    bool IsOk() const;  // Checks if the values are currently OK.
    int sample_rate_hz = 48000;
    int num_channels = 1;
  };
  static std::optional<Config> SdpToConfig(const SdpAudioFormat& audio_format);
  static void AppendSupportedDecoders(std::vector<AudioCodecSpec>* specs);

  static std::unique_ptr<AudioDecoder> MakeAudioDecoder(const Environment& env,
                                                        Config config);
  static std::unique_ptr<AudioDecoder> MakeAudioDecoder(
      const Environment& env,
      Config config,
      std::optional<AudioCodecPairId> /*codec_pair_id*/) {
    return MakeAudioDecoder(env, config);
  }
};

}  // namespace webrtc

#endif  // API_AUDIO_CODECS_OPUS_AUDIO_DECODER_OPUS_H_
