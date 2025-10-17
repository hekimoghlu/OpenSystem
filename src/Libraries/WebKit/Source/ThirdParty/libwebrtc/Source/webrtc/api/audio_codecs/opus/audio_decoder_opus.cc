/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#include "api/audio_codecs/opus/audio_decoder_opus.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "api/audio_codecs/audio_codec_pair_id.h"
#include "api/audio_codecs/audio_decoder.h"
#include "api/audio_codecs/audio_format.h"
#include "modules/audio_coding/codecs/opus/audio_decoder_opus.h"
#include "rtc_base/checks.h"

namespace webrtc {

bool AudioDecoderOpus::Config::IsOk() const {
  if (sample_rate_hz != 16000 && sample_rate_hz != 48000) {
    // Unsupported sample rate. (libopus supports a few other rates as
    // well; we can add support for them when needed.)
    return false;
  }
  if (num_channels != 1 && num_channels != 2) {
    return false;
  }
  return true;
}

std::optional<AudioDecoderOpus::Config> AudioDecoderOpus::SdpToConfig(
    const SdpAudioFormat& format) {
  const auto num_channels = [&]() -> std::optional<int> {
    auto stereo = format.parameters.find("stereo");
    if (stereo != format.parameters.end()) {
      if (stereo->second == "0") {
        return 1;
      } else if (stereo->second == "1") {
        return 2;
      } else {
        return std::nullopt;  // Bad stereo parameter.
      }
    }
    return 1;  // Default to mono.
  }();
  if (absl::EqualsIgnoreCase(format.name, "opus") &&
      format.clockrate_hz == 48000 && format.num_channels == 2 &&
      num_channels) {
    Config config;
    config.num_channels = *num_channels;
    if (!config.IsOk()) {
      RTC_DCHECK_NOTREACHED();
      return std::nullopt;
    }
    return config;
  } else {
    return std::nullopt;
  }
}

void AudioDecoderOpus::AppendSupportedDecoders(
    std::vector<AudioCodecSpec>* specs) {
  AudioCodecInfo opus_info{48000, 1, 64000, 6000, 510000};
  opus_info.allow_comfort_noise = false;
  opus_info.supports_network_adaption = true;
  SdpAudioFormat opus_format(
      {"opus", 48000, 2, {{"minptime", "10"}, {"useinbandfec", "1"}}});
  specs->push_back({std::move(opus_format), opus_info});
}

std::unique_ptr<AudioDecoder> AudioDecoderOpus::MakeAudioDecoder(
    const Environment& env,
    Config config) {
  if (!config.IsOk()) {
    RTC_DCHECK_NOTREACHED();
    return nullptr;
  }
  return std::make_unique<AudioDecoderOpusImpl>(
      env.field_trials(), config.num_channels, config.sample_rate_hz);
}

}  // namespace webrtc
