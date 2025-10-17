/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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
#include "api/audio_codecs/opus_audio_encoder_factory.h"

#include <memory>
#include <optional>
#include <vector>

#include "api/audio_codecs/audio_codec_pair_id.h"
#include "api/audio_codecs/audio_encoder.h"
#include "api/audio_codecs/audio_encoder_factory.h"
#include "api/audio_codecs/audio_encoder_factory_template.h"
#include "api/audio_codecs/audio_format.h"
#include "api/audio_codecs/opus/audio_encoder_multi_channel_opus.h"
#include "api/audio_codecs/opus/audio_encoder_opus.h"
#include "api/field_trials_view.h"
#include "api/scoped_refptr.h"

namespace webrtc {
namespace {

// Modify an audio encoder to not advertise support for anything.
template <typename T>
struct NotAdvertised {
  using Config = typename T::Config;
  static std::optional<Config> SdpToConfig(const SdpAudioFormat& audio_format) {
    return T::SdpToConfig(audio_format);
  }
  static void AppendSupportedEncoders(
      std::vector<AudioCodecSpec>* /* specs */) {
    // Don't advertise support for anything.
  }
  static AudioCodecInfo QueryAudioEncoder(const Config& config) {
    return T::QueryAudioEncoder(config);
  }
  static std::unique_ptr<AudioEncoder> MakeAudioEncoder(
      const Config& config,
      int payload_type,
      std::optional<AudioCodecPairId> codec_pair_id = std::nullopt,
      const FieldTrialsView* field_trials = nullptr) {
    return T::MakeAudioEncoder(config, payload_type, codec_pair_id,
                               field_trials);
  }
};

}  // namespace

rtc::scoped_refptr<AudioEncoderFactory> CreateOpusAudioEncoderFactory() {
  return CreateAudioEncoderFactory<
      AudioEncoderOpus, NotAdvertised<AudioEncoderMultiChannelOpus>>();
}

}  // namespace webrtc
