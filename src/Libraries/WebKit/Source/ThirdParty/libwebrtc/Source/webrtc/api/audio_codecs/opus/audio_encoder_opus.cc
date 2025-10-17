/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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
#include "api/audio_codecs/opus/audio_encoder_opus.h"

#include <memory>
#include <optional>
#include <vector>

#include "api/audio_codecs/audio_codec_pair_id.h"
#include "api/audio_codecs/audio_encoder.h"
#include "api/audio_codecs/audio_encoder_factory.h"
#include "api/audio_codecs/audio_format.h"
#include "api/audio_codecs/opus/audio_encoder_opus_config.h"
#include "modules/audio_coding/codecs/opus/audio_encoder_opus.h"
#include "rtc_base/checks.h"

namespace webrtc {

std::optional<AudioEncoderOpusConfig> AudioEncoderOpus::SdpToConfig(
    const SdpAudioFormat& format) {
  return AudioEncoderOpusImpl::SdpToConfig(format);
}

void AudioEncoderOpus::AppendSupportedEncoders(
    std::vector<AudioCodecSpec>* specs) {
  AudioEncoderOpusImpl::AppendSupportedEncoders(specs);
}

AudioCodecInfo AudioEncoderOpus::QueryAudioEncoder(
    const AudioEncoderOpusConfig& config) {
  return AudioEncoderOpusImpl::QueryAudioEncoder(config);
}

std::unique_ptr<AudioEncoder> AudioEncoderOpus::MakeAudioEncoder(
    const Environment& env,
    const AudioEncoderOpusConfig& config,
    const AudioEncoderFactory::Options& options) {
  if (!config.IsOk()) {
    RTC_DCHECK_NOTREACHED();
    return nullptr;
  }
  return std::make_unique<AudioEncoderOpusImpl>(env, config,
                                                options.payload_type);
}

}  // namespace webrtc
