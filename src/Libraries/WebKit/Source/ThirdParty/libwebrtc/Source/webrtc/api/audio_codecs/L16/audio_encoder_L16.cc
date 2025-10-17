/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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
#include "api/audio_codecs/L16/audio_encoder_L16.h"

#include <stddef.h>

#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "api/audio_codecs/audio_codec_pair_id.h"
#include "api/audio_codecs/audio_encoder.h"
#include "api/audio_codecs/audio_format.h"
#include "api/field_trials_view.h"
#include "modules/audio_coding/codecs/pcm16b/audio_encoder_pcm16b.h"
#include "modules/audio_coding/codecs/pcm16b/pcm16b_common.h"
#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"
#include "rtc_base/numerics/safe_minmax.h"
#include "rtc_base/string_to_number.h"

namespace webrtc {

std::optional<AudioEncoderL16::Config> AudioEncoderL16::SdpToConfig(
    const SdpAudioFormat& format) {
  if (!rtc::IsValueInRangeForNumericType<int>(format.num_channels)) {
    RTC_DCHECK_NOTREACHED();
    return std::nullopt;
  }
  Config config;
  config.sample_rate_hz = format.clockrate_hz;
  config.num_channels = rtc::dchecked_cast<int>(format.num_channels);
  auto ptime_iter = format.parameters.find("ptime");
  if (ptime_iter != format.parameters.end()) {
    const auto ptime = rtc::StringToNumber<int>(ptime_iter->second);
    if (ptime && *ptime > 0) {
      config.frame_size_ms = rtc::SafeClamp(10 * (*ptime / 10), 10, 60);
    }
  }
  if (absl::EqualsIgnoreCase(format.name, "L16") && config.IsOk()) {
    return config;
  }
  return std::nullopt;
}

void AudioEncoderL16::AppendSupportedEncoders(
    std::vector<AudioCodecSpec>* specs) {
  Pcm16BAppendSupportedCodecSpecs(specs);
}

AudioCodecInfo AudioEncoderL16::QueryAudioEncoder(
    const AudioEncoderL16::Config& config) {
  RTC_DCHECK(config.IsOk());
  return {config.sample_rate_hz,
          rtc::dchecked_cast<size_t>(config.num_channels),
          config.sample_rate_hz * config.num_channels * 16};
}

std::unique_ptr<AudioEncoder> AudioEncoderL16::MakeAudioEncoder(
    const AudioEncoderL16::Config& config,
    int payload_type,
    std::optional<AudioCodecPairId> /*codec_pair_id*/,
    const FieldTrialsView* /* field_trials */) {
  AudioEncoderPcm16B::Config c;
  c.sample_rate_hz = config.sample_rate_hz;
  c.num_channels = config.num_channels;
  c.frame_size_ms = config.frame_size_ms;
  c.payload_type = payload_type;
  if (!config.IsOk()) {
    RTC_DCHECK_NOTREACHED();
    return nullptr;
  }
  return std::make_unique<AudioEncoderPcm16B>(c);
}

}  // namespace webrtc
