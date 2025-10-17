/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_ACCELERATE_H_
#define MODULES_AUDIO_CODING_NETEQ_ACCELERATE_H_

#include <stddef.h>
#include <stdint.h>

#include "modules/audio_coding/neteq/time_stretch.h"

namespace webrtc {

class AudioMultiVector;
class BackgroundNoise;

// This class implements the Accelerate operation. Most of the work is done
// in the base class TimeStretch, which is shared with the PreemptiveExpand
// operation. In the Accelerate class, the operations that are specific to
// Accelerate are implemented.
class Accelerate : public TimeStretch {
 public:
  Accelerate(int sample_rate_hz,
             size_t num_channels,
             const BackgroundNoise& background_noise)
      : TimeStretch(sample_rate_hz, num_channels, background_noise) {}

  Accelerate(const Accelerate&) = delete;
  Accelerate& operator=(const Accelerate&) = delete;

  // This method performs the actual Accelerate operation. The samples are
  // read from `input`, of length `input_length` elements, and are written to
  // `output`. The number of samples removed through time-stretching is
  // is provided in the output `length_change_samples`. The method returns
  // the outcome of the operation as an enumerator value. If `fast_accelerate`
  // is true, the algorithm will relax the requirements on finding strong
  // correlations, and may remove multiple pitch periods if possible.
  ReturnCodes Process(const int16_t* input,
                      size_t input_length,
                      bool fast_accelerate,
                      AudioMultiVector* output,
                      size_t* length_change_samples);

 protected:
  // Sets the parameters `best_correlation` and `peak_index` to suitable
  // values when the signal contains no active speech.
  void SetParametersForPassiveSpeech(size_t len,
                                     int16_t* best_correlation,
                                     size_t* peak_index) const override;

  // Checks the criteria for performing the time-stretching operation and,
  // if possible, performs the time-stretching.
  ReturnCodes CheckCriteriaAndStretch(const int16_t* input,
                                      size_t input_length,
                                      size_t peak_index,
                                      int16_t best_correlation,
                                      bool active_speech,
                                      bool fast_mode,
                                      AudioMultiVector* output) const override;
};

struct AccelerateFactory {
  AccelerateFactory() {}
  virtual ~AccelerateFactory() {}

  virtual Accelerate* Create(int sample_rate_hz,
                             size_t num_channels,
                             const BackgroundNoise& background_noise) const;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_ACCELERATE_H_
