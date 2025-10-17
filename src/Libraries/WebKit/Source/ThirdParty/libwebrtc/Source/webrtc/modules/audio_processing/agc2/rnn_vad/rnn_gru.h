/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_RNN_GRU_H_
#define MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_RNN_GRU_H_

#include <array>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "modules/audio_processing/agc2/cpu_features.h"
#include "modules/audio_processing/agc2/rnn_vad/vector_math.h"

namespace webrtc {
namespace rnn_vad {

// Maximum number of units for a GRU layer.
constexpr int kGruLayerMaxUnits = 24;

// Recurrent layer with gated recurrent units (GRUs) with sigmoid and ReLU as
// activation functions for the update/reset and output gates respectively.
class GatedRecurrentLayer {
 public:
  // Ctor. `output_size` cannot be greater than `kGruLayerMaxUnits`.
  GatedRecurrentLayer(int input_size,
                      int output_size,
                      rtc::ArrayView<const int8_t> bias,
                      rtc::ArrayView<const int8_t> weights,
                      rtc::ArrayView<const int8_t> recurrent_weights,
                      const AvailableCpuFeatures& cpu_features,
                      absl::string_view layer_name);
  GatedRecurrentLayer(const GatedRecurrentLayer&) = delete;
  GatedRecurrentLayer& operator=(const GatedRecurrentLayer&) = delete;
  ~GatedRecurrentLayer();

  // Returns the size of the input vector.
  int input_size() const { return input_size_; }
  // Returns the pointer to the first element of the output buffer.
  const float* data() const { return state_.data(); }
  // Returns the size of the output buffer.
  int size() const { return output_size_; }

  // Resets the GRU state.
  void Reset();
  // Computes the recurrent layer output and updates the status.
  void ComputeOutput(rtc::ArrayView<const float> input);

 private:
  const int input_size_;
  const int output_size_;
  const std::vector<float> bias_;
  const std::vector<float> weights_;
  const std::vector<float> recurrent_weights_;
  const VectorMath vector_math_;
  // Over-allocated array with size equal to `output_size_`.
  std::array<float, kGruLayerMaxUnits> state_;
};

}  // namespace rnn_vad
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_RNN_GRU_H_
