/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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
#include "api/audio_codecs/g722/audio_decoder_g722.h"

#include <memory>
#include <optional>
#include <vector>

#include "absl/strings/match.h"
#include "api/audio_codecs/audio_codec_pair_id.h"
#include "api/audio_codecs/audio_decoder.h"
#include "api/audio_codecs/audio_format.h"
#include "api/field_trials_view.h"
#include "modules/audio_coding/codecs/g722/audio_decoder_g722.h"
#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"

namespace webrtc {

std::optional<AudioDecoderG722::Config> AudioDecoderG722::SdpToConfig(
    const SdpAudioFormat& format) {
  if (absl::EqualsIgnoreCase(format.name, "G722") &&
      format.clockrate_hz == 8000 &&
      (format.num_channels == 1 || format.num_channels == 2)) {
    return Config{rtc::dchecked_cast<int>(format.num_channels)};
  }
  return std::nullopt;
}

void AudioDecoderG722::AppendSupportedDecoders(
    std::vector<AudioCodecSpec>* specs) {
  specs->push_back({{"G722", 8000, 1}, {16000, 1, 64000}});
}

std::unique_ptr<AudioDecoder> AudioDecoderG722::MakeAudioDecoder(
    Config config,
    std::optional<AudioCodecPairId> /*codec_pair_id*/,
    const FieldTrialsView* /* field_trials */) {
  if (!config.IsOk()) {
    RTC_DCHECK_NOTREACHED();
    return nullptr;
  }
  switch (config.num_channels) {
    case 1:
      return std::make_unique<AudioDecoderG722Impl>();
    case 2:
      return std::make_unique<AudioDecoderG722StereoImpl>();
    default:
      RTC_DCHECK_NOTREACHED();
      return nullptr;
  }
}

}  // namespace webrtc
