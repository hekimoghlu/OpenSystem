/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_PREEMPTIVE_EXPAND_H_
#define MODULES_AUDIO_CODING_NETEQ_PREEMPTIVE_EXPAND_H_

#include <stddef.h>
#include <stdint.h>

#include "modules/audio_coding/neteq/time_stretch.h"

namespace webrtc {

class AudioMultiVector;
class BackgroundNoise;

// This class implements the PreemptiveExpand operation. Most of the work is
// done in the base class TimeStretch, which is shared with the Accelerate
// operation. In the PreemptiveExpand class, the operations that are specific to
// PreemptiveExpand are implemented.
class PreemptiveExpand : public TimeStretch {
 public:
  PreemptiveExpand(int sample_rate_hz,
                   size_t num_channels,
                   const BackgroundNoise& background_noise,
                   size_t overlap_samples)
      : TimeStretch(sample_rate_hz, num_channels, background_noise),
        old_data_length_per_channel_(0),
        overlap_samples_(overlap_samples) {}

  PreemptiveExpand(const PreemptiveExpand&) = delete;
  PreemptiveExpand& operator=(const PreemptiveExpand&) = delete;

  // This method performs the actual PreemptiveExpand operation. The samples are
  // read from `input`, of length `input_length` elements, and are written to
  // `output`. The number of samples added through time-stretching is
  // is provided in the output `length_change_samples`. The method returns
  // the outcome of the operation as an enumerator value.
  ReturnCodes Process(const int16_t* pw16_decoded,
                      size_t len,
                      size_t old_data_len,
                      AudioMultiVector* output,
                      size_t* length_change_samples);

 protected:
  // Sets the parameters `best_correlation` and `peak_index` to suitable
  // values when the signal contains no active speech.
  void SetParametersForPassiveSpeech(size_t input_length,
                                     int16_t* best_correlation,
                                     size_t* peak_index) const override;

  // Checks the criteria for performing the time-stretching operation and,
  // if possible, performs the time-stretching.
  ReturnCodes CheckCriteriaAndStretch(const int16_t* input,
                                      size_t input_length,
                                      size_t peak_index,
                                      int16_t best_correlation,
                                      bool active_speech,
                                      bool /*fast_mode*/,
                                      AudioMultiVector* output) const override;

 private:
  size_t old_data_length_per_channel_;
  size_t overlap_samples_;
};

struct PreemptiveExpandFactory {
  PreemptiveExpandFactory() {}
  virtual ~PreemptiveExpandFactory() {}

  virtual PreemptiveExpand* Create(int sample_rate_hz,
                                   size_t num_channels,
                                   const BackgroundNoise& background_noise,
                                   size_t overlap_samples) const;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_PREEMPTIVE_EXPAND_H_
