/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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
#include "api/audio_codecs/g711/audio_decoder_g711.h"

#include <initializer_list>
#include <memory>
#include <optional>
#include <vector>

#include "absl/strings/match.h"
#include "api/audio_codecs/audio_codec_pair_id.h"
#include "api/audio_codecs/audio_decoder.h"
#include "api/audio_codecs/audio_format.h"
#include "api/field_trials_view.h"
#include "modules/audio_coding/codecs/g711/audio_decoder_pcm.h"
#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"

namespace webrtc {

std::optional<AudioDecoderG711::Config> AudioDecoderG711::SdpToConfig(
    const SdpAudioFormat& format) {
  const bool is_pcmu = absl::EqualsIgnoreCase(format.name, "PCMU");
  const bool is_pcma = absl::EqualsIgnoreCase(format.name, "PCMA");
  if (format.clockrate_hz == 8000 && format.num_channels >= 1 &&
      (is_pcmu || is_pcma)) {
    Config config;
    config.type = is_pcmu ? Config::Type::kPcmU : Config::Type::kPcmA;
    config.num_channels = rtc::dchecked_cast<int>(format.num_channels);
    if (!config.IsOk()) {
      RTC_DCHECK_NOTREACHED();
      return std::nullopt;
    }
    return config;
  } else {
    return std::nullopt;
  }
}

void AudioDecoderG711::AppendSupportedDecoders(
    std::vector<AudioCodecSpec>* specs) {
  for (const char* type : {"PCMU", "PCMA"}) {
    specs->push_back({{type, 8000, 1}, {8000, 1, 64000}});
  }
}

std::unique_ptr<AudioDecoder> AudioDecoderG711::MakeAudioDecoder(
    const Config& config,
    std::optional<AudioCodecPairId> /*codec_pair_id*/,
    const FieldTrialsView* /* field_trials */) {
  if (!config.IsOk()) {
    RTC_DCHECK_NOTREACHED();
    return nullptr;
  }
  switch (config.type) {
    case Config::Type::kPcmU:
      return std::make_unique<AudioDecoderPcmU>(config.num_channels);
    case Config::Type::kPcmA:
      return std::make_unique<AudioDecoderPcmA>(config.num_channels);
    default:
      RTC_DCHECK_NOTREACHED();
      return nullptr;
  }
}

}  // namespace webrtc
