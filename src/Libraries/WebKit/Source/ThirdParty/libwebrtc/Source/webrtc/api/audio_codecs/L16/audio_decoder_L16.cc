/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
#include "api/audio_codecs/L16/audio_decoder_L16.h"

#include <memory>
#include <optional>
#include <vector>

#include "absl/strings/match.h"
#include "api/audio_codecs/audio_codec_pair_id.h"
#include "api/audio_codecs/audio_decoder.h"
#include "api/audio_codecs/audio_format.h"
#include "api/field_trials_view.h"
#include "modules/audio_coding/codecs/pcm16b/audio_decoder_pcm16b.h"
#include "modules/audio_coding/codecs/pcm16b/pcm16b_common.h"
#include "rtc_base/numerics/safe_conversions.h"

namespace webrtc {

std::optional<AudioDecoderL16::Config> AudioDecoderL16::SdpToConfig(
    const SdpAudioFormat& format) {
  Config config;
  config.sample_rate_hz = format.clockrate_hz;
  config.num_channels = rtc::checked_cast<int>(format.num_channels);
  if (absl::EqualsIgnoreCase(format.name, "L16") && config.IsOk()) {
    return config;
  }
  return std::nullopt;
}

void AudioDecoderL16::AppendSupportedDecoders(
    std::vector<AudioCodecSpec>* specs) {
  Pcm16BAppendSupportedCodecSpecs(specs);
}

std::unique_ptr<AudioDecoder> AudioDecoderL16::MakeAudioDecoder(
    const Config& config,
    std::optional<AudioCodecPairId> /*codec_pair_id*/,
    const FieldTrialsView* /* field_trials */) {
  if (!config.IsOk()) {
    return nullptr;
  }
  return std::make_unique<AudioDecoderPcm16B>(config.sample_rate_hz,
                                              config.num_channels);
}

}  // namespace webrtc
