/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 29, 2022.
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
#ifndef API_AUDIO_CODECS_G711_AUDIO_ENCODER_G711_H_
#define API_AUDIO_CODECS_G711_AUDIO_ENCODER_G711_H_

#include <memory>
#include <optional>
#include <vector>

#include "api/audio_codecs/audio_codec_pair_id.h"
#include "api/audio_codecs/audio_encoder.h"
#include "api/audio_codecs/audio_format.h"
#include "api/field_trials_view.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// G711 encoder API for use as a template parameter to
// CreateAudioEncoderFactory<...>().
struct RTC_EXPORT AudioEncoderG711 {
  struct Config {
    enum class Type { kPcmU, kPcmA };
    bool IsOk() const {
      return (type == Type::kPcmU || type == Type::kPcmA) &&
             frame_size_ms > 0 && frame_size_ms % 10 == 0 &&
             num_channels >= 1 &&
             num_channels <= AudioEncoder::kMaxNumberOfChannels;
    }
    Type type = Type::kPcmU;
    int num_channels = 1;
    int frame_size_ms = 20;
  };
  static std::optional<AudioEncoderG711::Config> SdpToConfig(
      const SdpAudioFormat& audio_format);
  static void AppendSupportedEncoders(std::vector<AudioCodecSpec>* specs);
  static AudioCodecInfo QueryAudioEncoder(const Config& config);
  static std::unique_ptr<AudioEncoder> MakeAudioEncoder(
      const Config& config,
      int payload_type,
      std::optional<AudioCodecPairId> codec_pair_id = std::nullopt,
      const FieldTrialsView* field_trials = nullptr);
};

}  // namespace webrtc

#endif  // API_AUDIO_CODECS_G711_AUDIO_ENCODER_G711_H_
