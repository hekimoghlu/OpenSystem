/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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
#ifndef AOM_AV1_ENCODER_ML_H_
#define AOM_AV1_ENCODER_ML_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "config/av1_rtcd.h"

#define NN_MAX_HIDDEN_LAYERS 10
#define NN_MAX_NODES_PER_LAYER 128

struct NN_CONFIG {
  int num_inputs;         // Number of input nodes, i.e. features.
  int num_outputs;        // Number of output nodes.
  int num_hidden_layers;  // Number of hidden layers, maximum 10.
  // Number of nodes for each hidden layer.
  int num_hidden_nodes[NN_MAX_HIDDEN_LAYERS];
  // Weight parameters, indexed by layer.
  const float *weights[NN_MAX_HIDDEN_LAYERS + 1];
  // Bias parameters, indexed by layer.
  const float *bias[NN_MAX_HIDDEN_LAYERS + 1];
};
// Typedef from struct NN_CONFIG to NN_CONFIG is in rtcd_defs

#if CONFIG_NN_V2
// Fully-connectedly layer configuration
struct FC_LAYER {
  const int num_inputs;   // Number of input nodes, i.e. features.
  const int num_outputs;  // Number of output nodes.

  float *weights;               // Weight parameters.
  float *bias;                  // Bias parameters.
  const ACTIVATION activation;  // Activation function.

  float *output;  // The output array.
  float *dY;      // Gradient of outputs
  float *dW;      // Gradient of weights.
  float *db;      // Gradient of bias
};

// NN configure structure V2
struct NN_CONFIG_V2 {
  const int num_hidden_layers;  // Number of hidden layers, max = 10.
  FC_LAYER layer[NN_MAX_HIDDEN_LAYERS + 1];  // The layer array
  const int num_logits;                      // Number of output nodes.
  float *logits;    // Raw prediction (same as output of final layer)
  const LOSS loss;  // Loss function
};

// Calculate prediction based on the given input features and neural net config.
// Assume there are no more than NN_MAX_NODES_PER_LAYER nodes in each hidden
// layer.
void av1_nn_predict_v2(const float *features, NN_CONFIG_V2 *nn_config,
                       int reduce_prec, float *output);
#endif  // CONFIG_NN_V2

// Applies the softmax normalization function to the input
// to get a valid probability distribution in the output:
// output[i] = exp(input[i]) / sum_{k \in [0,n)}(exp(input[k]))
void av1_nn_softmax(const float *input, float *output, int n);

// A faster but less accurate version of av1_nn_softmax(input, output, 16)
void av1_nn_fast_softmax_16_c(const float *input, float *output);

// Applies a precision reduction to output of av1_nn_predict to prevent
// mismatches between C and SIMD implementations.
void av1_nn_output_prec_reduce(float *const output, int num_output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_ML_H_
