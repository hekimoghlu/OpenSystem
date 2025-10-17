/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
#ifndef MODULES_AUDIO_CODING_CODECS_CNG_AUDIO_ENCODER_CNG_H_
#define MODULES_AUDIO_CODING_CODECS_CNG_AUDIO_ENCODER_CNG_H_

#include <stddef.h>

#include <memory>

#include "api/audio_codecs/audio_encoder.h"
#include "common_audio/vad/include/vad.h"

namespace webrtc {

struct AudioEncoderCngConfig {
  // Moveable, not copyable.
  AudioEncoderCngConfig();
  AudioEncoderCngConfig(AudioEncoderCngConfig&&);
  ~AudioEncoderCngConfig();

  bool IsOk() const;

  size_t num_channels = 1;
  int payload_type = 13;
  std::unique_ptr<AudioEncoder> speech_encoder;
  Vad::Aggressiveness vad_mode = Vad::kVadNormal;
  int sid_frame_interval_ms = 100;
  int num_cng_coefficients = 8;
  // The Vad pointer is mainly for testing. If a NULL pointer is passed, the
  // AudioEncoderCng creates (and destroys) a Vad object internally. If an
  // object is passed, the AudioEncoderCng assumes ownership of the Vad
  // object.
  Vad* vad = nullptr;
};

std::unique_ptr<AudioEncoder> CreateComfortNoiseEncoder(
    AudioEncoderCngConfig&& config);

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_CODECS_CNG_AUDIO_ENCODER_CNG_H_
