/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_RNN_H_
#define MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_RNN_H_

#include <stddef.h>
#include <sys/types.h>

#include <array>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/agc2/cpu_features.h"
#include "modules/audio_processing/agc2/rnn_vad/common.h"
#include "modules/audio_processing/agc2/rnn_vad/rnn_fc.h"
#include "modules/audio_processing/agc2/rnn_vad/rnn_gru.h"

namespace webrtc {
namespace rnn_vad {

// Recurrent network with hard-coded architecture and weights for voice activity
// detection.
class RnnVad {
 public:
  explicit RnnVad(const AvailableCpuFeatures& cpu_features);
  RnnVad(const RnnVad&) = delete;
  RnnVad& operator=(const RnnVad&) = delete;
  ~RnnVad();
  void Reset();
  // Observes `feature_vector` and `is_silence`, updates the RNN and returns the
  // current voice probability.
  float ComputeVadProbability(
      rtc::ArrayView<const float, kFeatureVectorSize> feature_vector,
      bool is_silence);

 private:
  FullyConnectedLayer input_;
  GatedRecurrentLayer hidden_;
  FullyConnectedLayer output_;
};

}  // namespace rnn_vad
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_RNN_H_
