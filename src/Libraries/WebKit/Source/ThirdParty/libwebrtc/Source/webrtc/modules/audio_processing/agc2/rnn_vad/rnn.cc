/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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
#include "modules/audio_processing/agc2/rnn_vad/rnn.h"

#include "rtc_base/checks.h"
#include "third_party/rnnoise/src/rnn_vad_weights.h"

namespace webrtc {
namespace rnn_vad {
namespace {

using ::rnnoise::kInputLayerInputSize;
static_assert(kFeatureVectorSize == kInputLayerInputSize, "");
using ::rnnoise::kInputDenseBias;
using ::rnnoise::kInputDenseWeights;
using ::rnnoise::kInputLayerOutputSize;
static_assert(kInputLayerOutputSize <= kFullyConnectedLayerMaxUnits, "");

using ::rnnoise::kHiddenGruBias;
using ::rnnoise::kHiddenGruRecurrentWeights;
using ::rnnoise::kHiddenGruWeights;
using ::rnnoise::kHiddenLayerOutputSize;
static_assert(kHiddenLayerOutputSize <= kGruLayerMaxUnits, "");

using ::rnnoise::kOutputDenseBias;
using ::rnnoise::kOutputDenseWeights;
using ::rnnoise::kOutputLayerOutputSize;
static_assert(kOutputLayerOutputSize <= kFullyConnectedLayerMaxUnits, "");

}  // namespace

RnnVad::RnnVad(const AvailableCpuFeatures& cpu_features)
    : input_(kInputLayerInputSize,
             kInputLayerOutputSize,
             kInputDenseBias,
             kInputDenseWeights,
             ActivationFunction::kTansigApproximated,
             cpu_features,
             /*layer_name=*/"FC1"),
      hidden_(kInputLayerOutputSize,
              kHiddenLayerOutputSize,
              kHiddenGruBias,
              kHiddenGruWeights,
              kHiddenGruRecurrentWeights,
              cpu_features,
              /*layer_name=*/"GRU1"),
      output_(kHiddenLayerOutputSize,
              kOutputLayerOutputSize,
              kOutputDenseBias,
              kOutputDenseWeights,
              ActivationFunction::kSigmoidApproximated,
              // The output layer is just 24x1. The unoptimized code is faster.
              NoAvailableCpuFeatures(),
              /*layer_name=*/"FC2") {
  // Input-output chaining size checks.
  RTC_DCHECK_EQ(input_.size(), hidden_.input_size())
      << "The input and the hidden layers sizes do not match.";
  RTC_DCHECK_EQ(hidden_.size(), output_.input_size())
      << "The hidden and the output layers sizes do not match.";
}

RnnVad::~RnnVad() = default;

void RnnVad::Reset() {
  hidden_.Reset();
}

float RnnVad::ComputeVadProbability(
    rtc::ArrayView<const float, kFeatureVectorSize> feature_vector,
    bool is_silence) {
  if (is_silence) {
    Reset();
    return 0.f;
  }
  input_.ComputeOutput(feature_vector);
  hidden_.ComputeOutput(input_);
  output_.ComputeOutput(hidden_);
  RTC_DCHECK_EQ(output_.size(), 1);
  return output_.data()[0];
}

}  // namespace rnn_vad
}  // namespace webrtc
