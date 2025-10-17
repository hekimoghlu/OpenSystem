/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_VAD_GMM_H_
#define MODULES_AUDIO_PROCESSING_VAD_GMM_H_

namespace webrtc {

// A structure that specifies a GMM.
// A GMM is formulated as
//  f(x) = w[0] * mixture[0] + w[1] * mixture[1] + ... +
//         w[num_mixtures - 1] * mixture[num_mixtures - 1];
// Where a 'mixture' is a Gaussian density.

struct GmmParameters {
  // weight[n] = log(w[n]) - `dimension`/2 * log(2*pi) - 1/2 * log(det(cov[n]));
  // where cov[n] is the covariance matrix of mixture n;
  const double* weight;
  // pointer to the first element of a `num_mixtures`x`dimension` matrix
  // where kth row is the mean of the kth mixture.
  const double* mean;
  // pointer to the first element of a `num_mixtures`x`dimension`x`dimension`
  // 3D-matrix, where the kth 2D-matrix is the inverse of the covariance
  // matrix of the kth mixture.
  const double* covar_inverse;
  // Dimensionality of the mixtures.
  int dimension;
  // number of the mixtures.
  int num_mixtures;
};

// Evaluate the given GMM, according to `gmm_parameters`, at the given point
// `x`. If the dimensionality of the given GMM is larger that the maximum
// acceptable dimension by the following function -1 is returned.
double EvaluateGmm(const double* x, const GmmParameters& gmm_parameters);

}  // namespace webrtc
#endif  // MODULES_AUDIO_PROCESSING_VAD_GMM_H_
