/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_ECHO_DETECTOR_NORMALIZED_COVARIANCE_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_ECHO_DETECTOR_NORMALIZED_COVARIANCE_ESTIMATOR_H_

namespace webrtc {

// This class iteratively estimates the normalized covariance between two
// signals.
class NormalizedCovarianceEstimator {
 public:
  void Update(float x,
              float x_mean,
              float x_var,
              float y,
              float y_mean,
              float y_var);
  // This function returns an estimate of the Pearson product-moment correlation
  // coefficient of the two signals.
  float normalized_cross_correlation() const {
    return normalized_cross_correlation_;
  }
  float covariance() const { return covariance_; }
  // This function resets the estimated values to zero.
  void Clear();

 private:
  float normalized_cross_correlation_ = 0.f;
  // Estimate of the covariance value.
  float covariance_ = 0.f;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_ECHO_DETECTOR_NORMALIZED_COVARIANCE_ESTIMATOR_H_
