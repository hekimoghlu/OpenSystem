/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_VAD_PITCH_BASED_VAD_H_
#define MODULES_AUDIO_PROCESSING_VAD_PITCH_BASED_VAD_H_

#include <memory>

#include "modules/audio_processing/vad/common.h"
#include "modules/audio_processing/vad/gmm.h"

namespace webrtc {

class VadCircularBuffer;

// Computes the probability of the input audio frame to be active given
// the corresponding pitch-gain and lag of the frame.
class PitchBasedVad {
 public:
  PitchBasedVad();
  ~PitchBasedVad();

  // Compute pitch-based voicing probability, given the features.
  //   features: a structure containing features required for computing voicing
  //             probabilities.
  //
  //   p_combined: an array which contains the combined activity probabilities
  //               computed prior to the call of this function. The method,
  //               then, computes the voicing probabilities and combine them
  //               with the given values. The result are returned in `p`.
  int VoicingProbability(const AudioFeatures& features, double* p_combined);

 private:
  int UpdatePrior(double p);

  // TODO(turajs): maybe defining this at a higher level (maybe enum) so that
  // all the code recognize it as "no-error."
  static const int kNoError = 0;

  GmmParameters noise_gmm_;
  GmmParameters voice_gmm_;

  double p_prior_;

  std::unique_ptr<VadCircularBuffer> circular_buffer_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_VAD_PITCH_BASED_VAD_H_
