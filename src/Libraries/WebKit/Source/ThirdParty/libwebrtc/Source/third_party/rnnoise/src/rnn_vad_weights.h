/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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

#ifndef THIRD_PARTY_RNNOISE_SRC_RNN_VAD_WEIGHTS_H_
#define THIRD_PARTY_RNNOISE_SRC_RNN_VAD_WEIGHTS_H_

#include <cstdint>
#include <cstring>

namespace rnnoise {

// Weights scaling factor.
const float kWeightsScale = 1.f / 256.f;

// Input layer (dense).
const size_t kInputLayerInputSize = 42;
const size_t kInputLayerOutputSize = 24;
const size_t kInputLayerWeights = kInputLayerInputSize * kInputLayerOutputSize;
extern const int8_t kInputDenseWeights[kInputLayerWeights];
extern const int8_t kInputDenseBias[kInputLayerOutputSize];

// Hidden layer (GRU).
const size_t kHiddenLayerOutputSize = 24;
const size_t kHiddenLayerWeights =
    3 * kInputLayerOutputSize * kHiddenLayerOutputSize;
const size_t kHiddenLayerBiases = 3 * kHiddenLayerOutputSize;
extern const int8_t kHiddenGruWeights[kHiddenLayerWeights];
extern const int8_t kHiddenGruRecurrentWeights[kHiddenLayerWeights];
extern const int8_t kHiddenGruBias[kHiddenLayerBiases];

// Output layer (dense).
const size_t kOutputLayerOutputSize = 1;
const size_t kOutputLayerWeights =
    kHiddenLayerOutputSize * kOutputLayerOutputSize;
extern const int8_t kOutputDenseWeights[kOutputLayerWeights];
extern const int8_t kOutputDenseBias[kOutputLayerOutputSize];

}  // namespace rnnoise

#endif  // THIRD_PARTY_RNNOISE_SRC_RNN_VAD_WEIGHTS_H_
