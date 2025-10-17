/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_DTMF_TONE_GENERATOR_H_
#define MODULES_AUDIO_CODING_NETEQ_DTMF_TONE_GENERATOR_H_

#include <stddef.h>
#include <stdint.h>

#include "modules/audio_coding/neteq/audio_multi_vector.h"

namespace webrtc {

// This class provides a generator for DTMF tones.
class DtmfToneGenerator {
 public:
  enum ReturnCodes {
    kNotInitialized = -1,
    kParameterError = -2,
  };

  DtmfToneGenerator();
  virtual ~DtmfToneGenerator() {}

  DtmfToneGenerator(const DtmfToneGenerator&) = delete;
  DtmfToneGenerator& operator=(const DtmfToneGenerator&) = delete;

  virtual int Init(int fs, int event, int attenuation);
  virtual void Reset();
  virtual int Generate(size_t num_samples, AudioMultiVector* output);
  virtual bool initialized() const;

 private:
  static const int kCoeff1[4][16];  // 1st oscillator model coefficient table.
  static const int kCoeff2[4][16];  // 2nd oscillator model coefficient table.
  static const int kInitValue1[4][16];  // Initialization for 1st oscillator.
  static const int kInitValue2[4][16];  // Initialization for 2nd oscillator.
  static const int kAmplitude[64];      // Amplitude for 0 through -63 dBm0.
  static const int16_t kAmpMultiplier = 23171;  // 3 dB attenuation (in Q15).

  bool initialized_;            // True if generator is initialized properly.
  int coeff1_;                  // 1st oscillator coefficient for this event.
  int coeff2_;                  // 2nd oscillator coefficient for this event.
  int amplitude_;               // Amplitude for this event.
  int16_t sample_history1_[2];  // Last 2 samples for the 1st oscillator.
  int16_t sample_history2_[2];  // Last 2 samples for the 2nd oscillator.
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_DTMF_TONE_GENERATOR_H_
