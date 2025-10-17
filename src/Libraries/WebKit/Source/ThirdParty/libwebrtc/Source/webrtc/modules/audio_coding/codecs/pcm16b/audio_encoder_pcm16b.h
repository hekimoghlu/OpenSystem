/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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
#ifndef MODULES_AUDIO_CODING_CODECS_PCM16B_AUDIO_ENCODER_PCM16B_H_
#define MODULES_AUDIO_CODING_CODECS_PCM16B_AUDIO_ENCODER_PCM16B_H_

#include "modules/audio_coding/codecs/g711/audio_encoder_pcm.h"

namespace webrtc {

class AudioEncoderPcm16B final : public AudioEncoderPcm {
 public:
  struct Config : public AudioEncoderPcm::Config {
   public:
    Config() : AudioEncoderPcm::Config(107), sample_rate_hz(8000) {}
    bool IsOk() const;

    int sample_rate_hz;
  };

  explicit AudioEncoderPcm16B(const Config& config)
      : AudioEncoderPcm(config, config.sample_rate_hz) {}

  AudioEncoderPcm16B(const AudioEncoderPcm16B&) = delete;
  AudioEncoderPcm16B& operator=(const AudioEncoderPcm16B&) = delete;

 protected:
  size_t EncodeCall(const int16_t* audio,
                    size_t input_len,
                    uint8_t* encoded) override;

  size_t BytesPerSample() const override;

  AudioEncoder::CodecType GetCodecType() const override;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_CODECS_PCM16B_AUDIO_ENCODER_PCM16B_H_
