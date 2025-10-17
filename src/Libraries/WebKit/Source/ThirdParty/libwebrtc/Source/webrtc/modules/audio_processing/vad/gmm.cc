/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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
#include "modules/audio_processing/vad/gmm.h"

#include <math.h>

namespace webrtc {

static const int kMaxDimension = 10;

static void RemoveMean(const double* in,
                       const double* mean_vec,
                       int dimension,
                       double* out) {
  for (int n = 0; n < dimension; ++n)
    out[n] = in[n] - mean_vec[n];
}

static double ComputeExponent(const double* in,
                              const double* covar_inv,
                              int dimension) {
  double q = 0;
  for (int i = 0; i < dimension; ++i) {
    double v = 0;
    for (int j = 0; j < dimension; j++)
      v += (*covar_inv++) * in[j];
    q += v * in[i];
  }
  q *= -0.5;
  return q;
}

double EvaluateGmm(const double* x, const GmmParameters& gmm_parameters) {
  if (gmm_parameters.dimension > kMaxDimension) {
    return -1;  // This is invalid pdf so the caller can check this.
  }
  double f = 0;
  double v[kMaxDimension];
  const double* mean_vec = gmm_parameters.mean;
  const double* covar_inv = gmm_parameters.covar_inverse;

  for (int n = 0; n < gmm_parameters.num_mixtures; n++) {
    RemoveMean(x, mean_vec, gmm_parameters.dimension, v);
    double q = ComputeExponent(v, covar_inv, gmm_parameters.dimension) +
               gmm_parameters.weight[n];
    f += exp(q);
    mean_vec += gmm_parameters.dimension;
    covar_inv += gmm_parameters.dimension * gmm_parameters.dimension;
  }
  return f;
}

}  // namespace webrtc
