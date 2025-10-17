/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#ifndef COMMON_AUDIO_VAD_INCLUDE_VAD_H_
#define COMMON_AUDIO_VAD_INCLUDE_VAD_H_

#include <memory>

#include "common_audio/vad/include/webrtc_vad.h"
#include "rtc_base/checks.h"

namespace webrtc {

class Vad {
 public:
  enum Aggressiveness {
    kVadNormal = 0,
    kVadLowBitrate = 1,
    kVadAggressive = 2,
    kVadVeryAggressive = 3
  };

  enum Activity { kPassive = 0, kActive = 1, kError = -1 };

  virtual ~Vad() = default;

  // Calculates a VAD decision for the given audio frame. Valid sample rates
  // are 8000, 16000, and 32000 Hz; the number of samples must be such that the
  // frame is 10, 20, or 30 ms long.
  virtual Activity VoiceActivity(const int16_t* audio,
                                 size_t num_samples,
                                 int sample_rate_hz) = 0;

  // Resets VAD state.
  virtual void Reset() = 0;
};

// Returns a Vad instance that's implemented on top of WebRtcVad.
std::unique_ptr<Vad> CreateVad(Vad::Aggressiveness aggressiveness);

}  // namespace webrtc

#endif  // COMMON_AUDIO_VAD_INCLUDE_VAD_H_
