/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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
#include "api/audio_codecs/g711/audio_encoder_g711.h"

#include <stddef.h>

#include <initializer_list>
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
#include "modules/audio_coding/codecs/g711/audio_encoder_pcm.h"
#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"
#include "rtc_base/numerics/safe_minmax.h"
#include "rtc_base/string_to_number.h"

namespace webrtc {

std::optional<AudioEncoderG711::Config> AudioEncoderG711::SdpToConfig(
    const SdpAudioFormat& format) {
  const bool is_pcmu = absl::EqualsIgnoreCase(format.name, "PCMU");
  const bool is_pcma = absl::EqualsIgnoreCase(format.name, "PCMA");
  if (format.clockrate_hz == 8000 && format.num_channels >= 1 &&
      (is_pcmu || is_pcma)) {
    Config config;
    config.type = is_pcmu ? Config::Type::kPcmU : Config::Type::kPcmA;
    config.num_channels = rtc::dchecked_cast<int>(format.num_channels);
    config.frame_size_ms = 20;
    auto ptime_iter = format.parameters.find("ptime");
    if (ptime_iter != format.parameters.end()) {
      const auto ptime = rtc::StringToNumber<int>(ptime_iter->second);
      if (ptime && *ptime > 0) {
        config.frame_size_ms = rtc::SafeClamp(10 * (*ptime / 10), 10, 60);
      }
    }
    if (!config.IsOk()) {
      RTC_DCHECK_NOTREACHED();
      return std::nullopt;
    }
    return config;
  } else {
    return std::nullopt;
  }
}

void AudioEncoderG711::AppendSupportedEncoders(
    std::vector<AudioCodecSpec>* specs) {
  for (const char* type : {"PCMU", "PCMA"}) {
    specs->push_back({{type, 8000, 1}, {8000, 1, 64000}});
  }
}

AudioCodecInfo AudioEncoderG711::QueryAudioEncoder(const Config& config) {
  RTC_DCHECK(config.IsOk());
  return {8000, rtc::dchecked_cast<size_t>(config.num_channels),
          64000 * config.num_channels};
}

std::unique_ptr<AudioEncoder> AudioEncoderG711::MakeAudioEncoder(
    const Config& config,
    int payload_type,
    std::optional<AudioCodecPairId> /*codec_pair_id*/,
    const FieldTrialsView* /* field_trials */) {
  if (!config.IsOk()) {
    RTC_DCHECK_NOTREACHED();
    return nullptr;
  }
  switch (config.type) {
    case Config::Type::kPcmU: {
      AudioEncoderPcmU::Config impl_config;
      impl_config.num_channels = config.num_channels;
      impl_config.frame_size_ms = config.frame_size_ms;
      impl_config.payload_type = payload_type;
      return std::make_unique<AudioEncoderPcmU>(impl_config);
    }
    case Config::Type::kPcmA: {
      AudioEncoderPcmA::Config impl_config;
      impl_config.num_channels = config.num_channels;
      impl_config.frame_size_ms = config.frame_size_ms;
      impl_config.payload_type = payload_type;
      return std::make_unique<AudioEncoderPcmA>(impl_config);
    }
    default: {
      RTC_DCHECK_NOTREACHED();
      return nullptr;
    }
  }
}

}  // namespace webrtc
